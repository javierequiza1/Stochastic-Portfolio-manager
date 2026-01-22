using JuMP, Gurobi, DataFrames, CSV, Plots

# ==============================================================================
# 1. CARGA DE DATOS (HÍBRIDA)
# ==============================================================================

# --- A) Escenarios Iniciales (5 Ramas) ---
# Asegúrate de que el archivo .csv está en la carpeta correcta
df_ini = CSV.read("escenarios_semestrales.csv", DataFrame)
activos = df_ini[:, 1] # Nombres de los activos

# Extraemos vectores de retornos para t=1
ret_MB = df_ini[:, 2] # Muy Bajista (10%)
ret_B  = df_ini[:, 3] # Bajista (20%)
ret_N  = df_ini[:, 4] # Neutral (40%)
ret_A  = df_ini[:, 5] # Alcista (20%)
ret_MA = df_ini[:, 6] # Muy Alcista (10%)

# --- B) Escenarios Futuros (2 Ramas Binarias Independientes) ---
df_fut = CSV.read("escenarios_futuros_binarios.csv", DataFrame)
ret_fut_B = df_fut[:, 2]
ret_fut_A = df_fut[:, 3]

# ==============================================================================
# 2. CONFIGURACIÓN DEL ÁRBOL
# ==============================================================================

T = 4 # Semestres
cap_inicial = 2000
cap_objetivo = 3000.0 

# Probabilidades de las 5 ramas iniciales
probs_iniciales = [0.10, 0.20, 0.40, 0.20, 0.10] 

# Cálculo de Hojas Totales (5 * 2^9)
n_hojas = 5 * 2^(T-1) 
println("Construyendo árbol con $n_hojas escenarios finales...")

# Generación del vector de Probabilidades Finales
probs_hojas = zeros(n_hojas)
bloque_inicial = Int(n_hojas / 5)

for i in 1:5
    p_hoja = probs_iniciales[i] * (0.5)^(T-1)
    idx_inicio = (i-1)*bloque_inicial + 1
    idx_fin    = i*bloque_inicial
    probs_hojas[idx_inicio:idx_fin] .= p_hoja
end

if !isapprox(sum(probs_hojas), 1.0, atol=1e-4)
    error("Las probabilidades no suman 100%. Suman: $(sum(probs_hojas))")
end

# ==============================================================================
# 3. MODELO DE OPTIMIZACIÓN
# ==============================================================================

model = Model(Gurobi.Optimizer)
set_silent(model)

# Variables
@variable(model, x[1:length(activos), 1:T, 1:n_hojas] >= 0)
@variable(model, Wealth[1:T+1, 1:n_hojas] >= 0)
@variable(model, q_neg[1:n_hojas] >= 0)

# --- RESTRICCIONES ---

# 1. Riqueza Inicial
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
        
        # A) NO ANTICIPATIVIDAD
        for n in rango
            if n != lider
                @constraint(model, x[:, t, n] .== x[:, t, lider])
            end
            
            # B) PRESUPUESTO
            @constraint(model, sum(x[:, t, n]) == Wealth[t, n])
            
            # C) LÍMITES DE INVERSIÓN
            for i in 1:length(activos)
                @constraint(model, x[i, t, n] <= 0.40 * Wealth[t, n])
            end
            
            # Max 20% en Cripto + Uranio (Activos 5, 6, 7)
            @constraint(model, sum(x[i, t, n] for i in [5,6,7]) <= 0.20 * Wealth[t, n])
        end

        # D) TRANSICIÓN DE RIQUEZA
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
            # Mitad 1: Bajista Futuro
            for n in rango[1:mitad]
                @constraint(model, Wealth[t+1, n] == sum(x[i, t, n] * (1 + ret_fut_B[i]) for i in 1:length(activos)))
            end
            # Mitad 2: Alcista Futuro
            for n in rango[mitad+1:end]
                @constraint(model, Wealth[t+1, n] == sum(x[i, t, n] * (1 + ret_fut_A[i]) for i in 1:length(activos)))
            end
        end
    end
end

# --- OBJETIVO ---
for n in 1:n_hojas
    @constraint(model, q_neg[n] >= cap_objetivo - Wealth[T+1, n])
end

@objective(model, Min, sum(q_neg[n] * probs_hojas[n] for n in 1:n_hojas))

println("Optimizando modelo híbrido...")
optimize!(model)

# ==============================================================================
# 4. RESULTADOS Y VISUALIZACIÓN
# ==============================================================================

if termination_status(model) == MOI.OPTIMAL
    println("\n" * "="^50)
    println(" PORTFOLIO ÓPTIMO INICIAL (T=0 -> T=1)")
    println("="^50)
    
    # HE QUITADO LA VARIABLE 'total_inv' PARA EVITAR EL ERROR DE JULIA
    for i in 1:length(activos)
        dinero = value(x[i, 1, 1])
        peso = dinero / cap_inicial
        
        if peso > 0.001
            println(rpad(activos[i], 15), ": ", lpad(round(peso*100, digits=2), 5), "%")
        end
    end
    
    # Métricas Finales
    vals_finales = value.(Wealth[T+1, :])
    esperanza = sum(vals_finales .* probs_hojas)
    prob_exito = sum(probs_hojas[vals_finales .>= cap_objetivo]) * 100
    
    println("-"^50)
    println("Riqueza Esperada Final: ", round(Int, esperanza), " €")
    println("Probabilidad de Lograr 40k: ", round(prob_exito, digits=1), "%")
    
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
        
        println("\n--- SEMESTRE $t $estado ---")
        println("Capital: ", round(Int, w_actual), " €  |  Rendimiento: ", round(rent*100, digits=2), "%")
        
        # Cartera
        invs = []
        for i in 1:length(activos)
            monto = value(x[i, t, idx_peor])
            if (monto/w_actual) > 0.01
                push!(invs, (activos[i], monto/w_actual))
            end
        end
        sort!(invs, by=x->x[2], rev=true)
        for (act, peso) in invs
            println("  > ", rpad(act, 10), ": ", round(peso*100, digits=1), "%")
        end
    end

    # --- GRÁFICO DE ABANICO ---
    println("\nGenerando gráfico...")
    plt = plot(title="Proyección Híbrida (5 Ramas -> Binario)", legend=false, xlabel="Semestres", ylabel="Capital (€)")
    
    # Seleccionamos 150 caminos representativos
    indices_plot = unique(Int.(round.(range(1, n_hojas, length=150))))
    
    for n in indices_plot
        camino = [value(Wealth[t, n]) for t in 1:T+1]
        col = camino[end] >= cap_objetivo ? :green : :red
        plot!(plt, 0:T, camino, color=col, alpha=0.15)
    end
    
    hline!(plt, [cap_objetivo], color=:blue, linestyle=:dash, linewidth=2, label="Meta")
    hline!(plt, [cap_inicial], color=:black, linestyle=:dot, label="Inicio")
    
    display(plt)

else
    println("El modelo no encontró solución óptima: ", termination_status(model))
end

# ==========================================================================
    # 5. ANÁLISIS DE RIESGO: DISTRIBUCIÓN DE RIQUEZA FINAL
    # ==========================================================================
    
    println("\nGenerando distribución de probabilidad final...")

    # 1. Preparar datos
    vals_finales = value.(Wealth[T+1, :])
    
    # 2. Calcular métricas de Riesgo (VaR y CVaR al 95%)
    # Creamos un DataFrame temporal para ordenar por riqueza y acumular probabilidad
    df_risk = DataFrame(Wealth = vals_finales, Prob = probs_hojas)
    sort!(df_risk, :Wealth) # Ordenamos del peor al mejor escenario
    df_risk.CumProb = cumsum(df_risk.Prob)
    
    # Value at Risk (VaR) 95%: El valor por debajo del cual cae el 5% de los peores escenarios
    idx_var = findfirst(x -> x >= 0.05, df_risk.CumProb)
    var_95 = df_risk.Wealth[idx_var]
    
    # Conditional VaR (CVaR): El promedio de pérdidas en ese peor 5% (La catástrofe media)
    # Re-normalizamos las probabilidades de esa cola para sacar la media
    tail_probs = df_risk.Prob[1:idx_var] ./ df_risk.CumProb[idx_var]
    cvar_95 = sum(df_risk.Wealth[1:idx_var] .* tail_probs)

    println("-"^50)
    println("MÉTRICAS DE RIESGO DE COLA (The Tail Risk):")
    println("VaR 95%  (Límite del peor 5%):  ", round(Int, var_95), " €")
    println("CVaR 95% (Media del desastre):  ", round(Int, cvar_95), " €")
    println("-"^50)

    # 3. Plot del Histograma
    # Usamos 'weights' porque no todos los escenarios tienen la misma probabilidad
    p_hist = histogram(vals_finales, weights=probs_hojas,
        bins=50, 
        label="Distribución de Escenarios",
        color=:skyblue,
        linecolor=:white,
        alpha=0.7,
        xlabel="Riqueza Final (€)",
        ylabel="Probabilidad",
        title="Distribución de Riqueza Final (T=10)",
        legend=:topright
    )

    # Líneas de Referencia
    vline!(p_hist, [cap_inicial], label="Capital Inicial (20k)", color=:red, linewidth=2)
    vline!(p_hist, [cap_objetivo], label="Objetivo (40k)", color=:green, linewidth=2, linestyle=:dash)
    vline!(p_hist, [var_95], label="VaR 5% ($(round(Int, var_95)))", color=:orange, linestyle=:dot, linewidth=2)

    display(p_hist)