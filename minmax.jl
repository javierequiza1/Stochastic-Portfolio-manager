using JuMP, Gurobi, DataFrames, CSV, Plots

# ==============================================================================
# 1. CARGA DE DATOS
# ==============================================================================

df_ini = CSV.read("escenarios_semestrales.csv", DataFrame)
activos = df_ini[:, 1] 

# Retornos T=1
ret_MB = df_ini[:, 2] 
ret_B  = df_ini[:, 3] 
ret_N  = df_ini[:, 4] 
ret_A  = df_ini[:, 5] 
ret_MA = df_ini[:, 6] 

# Retornos Futuros
df_fut = CSV.read("escenarios_futuros_binarios.csv", DataFrame)
ret_fut_B = df_fut[:, 2]
ret_fut_A = df_fut[:, 3]

# ==============================================================================
# 2. CONFIGURACIÓN DEL ÁRBOL
# ==============================================================================

T = 6
cap_inicial = 2000.0
cap_objetivo = 4000.0 

probs_iniciales = [0.10, 0.20, 0.40, 0.20, 0.10] 
n_hojas = 5 * 2^(T-1) 
println("Construyendo árbol Min-Max con $n_hojas escenarios...")

# Probabilidades (necesarias solo para los gráficos finales, no para la optimización MinMax)
probs_hojas = zeros(n_hojas)
bloque_inicial = Int(n_hojas / 5)

for i in 1:5
    p_hoja = probs_iniciales[i] * (0.5)^(T-1)
    idx_inicio = (i-1)*bloque_inicial + 1
    idx_fin    = i*bloque_inicial
    probs_hojas[idx_inicio:idx_fin] .= p_hoja
end

# ==============================================================================
# 3. MODELO DE OPTIMIZACIÓN (ENFOQUE MIN-MAX)
# ==============================================================================

model = Model(Gurobi.Optimizer)
set_silent(model)

# Variables
@variable(model, x[1:length(activos), 1:T, 1:n_hojas] >= 0)
@variable(model, Wealth[1:T+1, 1:n_hojas] >= 0)
@variable(model, q_neg[1:n_hojas] >= 0) # Déficit de cada hoja

# NUEVA VARIABLE PARA MIN-MAX: El peor déficit posible
@variable(model, max_deficit >= 0)

# --- RESTRICCIONES (Estructura del Árbol igual que antes) ---

@constraint(model, [n=1:n_hojas], Wealth[1, n] == cap_inicial)

for t in 1:T
    if t == 1
        grupos = [1:n_hojas]
    else
        n_grupos = 5 * 2^(t-2)
        tam_grupo = Int(n_hojas / n_grupos)
        grupos = [ (g-1)*tam_grupo+1 : g*tam_grupo for g in 1:n_grupos ]
    end

    for rango in grupos
        lider = first(rango)
        for n in rango
            if n != lider
                @constraint(model, x[:, t, n] .== x[:, t, lider])
            end
            @constraint(model, sum(x[:, t, n]) == Wealth[t, n])
            
            # Constraints de negocio
            for i in 1:length(activos)
                @constraint(model, x[i, t, n] <= 0.40 * Wealth[t, n])
            end
            @constraint(model, sum(x[i, t, n] for i in [5,6,7]) <= 0.20 * Wealth[t, n])
        end

        if t == 1
            bloque_hijos = Int(n_hojas / 5)
            lista_rets = [ret_MB, ret_B, ret_N, ret_A, ret_MA]
            for rama in 1:5
                inicio = (rama-1)*bloque_hijos + 1
                fin    = rama*bloque_hijos
                for h in inicio:fin
                    @constraint(model, Wealth[t+1, h] == sum(x[i, t, 1] * (1 + lista_rets[rama][i]) for i in 1:length(activos)))
                end
            end
        else
            mitad = Int(length(rango) / 2)
            for n in rango[1:mitad]
                @constraint(model, Wealth[t+1, n] == sum(x[i, t, n] * (1 + ret_fut_B[i]) for i in 1:length(activos)))
            end
            for n in rango[mitad+1:end]
                @constraint(model, Wealth[t+1, n] == sum(x[i, t, n] * (1 + ret_fut_A[i]) for i in 1:length(activos)))
            end
        end
    end
end

# --- OBJETIVO MIN-MAX ---

# 1. Definimos el déficit de cada escenario individualmente
for n in 1:n_hojas
    @constraint(model, q_neg[n] >= cap_objetivo - Wealth[T+1, n])
end

# 2. La variable 'max_deficit' debe ser mayor o igual que EL PEOR de todos los déficits
for n in 1:n_hojas
    @constraint(model, max_deficit >= q_neg[n])
end

# 3. Minimizamos ese techo (Minimizar el máximo dolor)
@objective(model, Min, max_deficit)

println("Optimizando estrategia Robusta (Min-Max)...")
optimize!(model)

# ==============================================================================
# 4. RESULTADOS Y VISUALIZACIÓN
# ==============================================================================

if termination_status(model) == MOI.OPTIMAL
    println("\n" * "="^50)
    println(" PORTFOLIO ROBUSTO (MIN-MAX)")
    println(" Estrategia: Protegerse del Peor Escenario Absoluto")
    println("="^50)
    
    for i in 1:length(activos)
        dinero = value(x[i, 1, 1])
        peso = dinero / cap_inicial
        if peso > 0.001
            println(rpad(activos[i], 15), ": ", lpad(round(peso*100, digits=2), 5), "%")
        end
    end
    
    vals_finales = value.(Wealth[T+1, :])
    esperanza = sum(vals_finales .* probs_hojas)
    prob_exito = sum(probs_hojas[vals_finales .>= cap_objetivo]) * 100
    
    println("-"^50)
    println("Riqueza Esperada Final: ", round(Int, esperanza), " €")
    println("Probabilidad de Lograr 40k: ", round(prob_exito, digits=1), "%")
    println("PEOR RIQUEZA POSIBLE: ", round(Int, minimum(vals_finales)), " € (El suelo garantizado)")
    
    # --- AUTOPSIA DEL PEOR CASO ---
    min_val, idx_peor = findmin(vals_finales)

    println("\n" * "="^60)
    println(" AUTOPSIA: EL PEOR CAMINO POSIBLE (Hoja #$idx_peor)")
    println(" Riqueza Final: ", round(Int, min_val), " €")
    println("="^60)

    for t in 1:T
        w_actual = value(Wealth[t, idx_peor])
        w_sig    = value(Wealth[t+1, idx_peor])
        rent     = (w_sig / w_actual) - 1
        estado = rent < -0.05 ? "[CRASH]" : (rent < 0 ? "[BAJADA]" : "[SUBIDA]")
        println("Semestre $t $estado: ", round(Int, w_actual), " €")
    end

    # --- GRÁFICO DE ABANICO ---
    println("\nGenerando gráfico de caminos...")
    plt = plot(title="Proyección Min-Max (Robusta)", legend=false, xlabel="Semestres", ylabel="Capital (€)")
    indices_plot = unique(Int.(round.(range(1, n_hojas, length=150))))
    for n in indices_plot
        camino = [value(Wealth[t, n]) for t in 1:T+1]
        col = camino[end] >= cap_objetivo ? :green : :red
        plot!(plt, 0:T, camino, color=col, alpha=0.15)
    end
    hline!(plt, [cap_objetivo], color=:blue, linestyle=:dash, linewidth=2, label="Meta")
    hline!(plt, [cap_inicial], color=:black, linestyle=:dot, label="Inicio")
    display(plt)

    # --- HISTOGRAMA ---
    println("\nGenerando distribución...")
    df_risk = DataFrame(Wealth = vals_finales, Prob = probs_hojas)
    sort!(df_risk, :Wealth)
    df_risk.CumProb = cumsum(df_risk.Prob)
    idx_var = findfirst(x -> x >= 0.05, df_risk.CumProb)
    var_95 = df_risk.Wealth[idx_var]
    
    p_hist = histogram(vals_finales, weights=probs_hojas,
        bins=50, label="Distribución", color=:orange, linecolor=:white, alpha=0.7,
        title="Distribución Min-Max (Compacta)", xlabel="Riqueza Final (€)")
    
    vline!(p_hist, [cap_inicial], color=:red, linewidth=2)
    vline!(p_hist, [cap_objetivo], color=:green, linestyle=:dash, linewidth=2)
    vline!(p_hist, [minimum(vals_finales)], label="Suelo Min-Max", color=:black, linewidth=2)
    
    display(p_hist)

else
    println("El modelo no encontró solución óptima: ", termination_status(model))
end