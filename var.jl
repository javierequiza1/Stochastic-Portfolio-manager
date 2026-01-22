using JuMP, Gurobi, DataFrames, CSV, Plots

# ==============================================================================
# 1. CARGA DE DATOS
# ==============================================================================
df_ini = CSV.read("escenarios_semestrales.csv", DataFrame)
activos = df_ini[:, 1] 
ret_MB = df_ini[:, 2]; ret_B = df_ini[:, 3]; ret_N = df_ini[:, 4] 
ret_A = df_ini[:, 5]; ret_MA = df_ini[:, 6] 

df_fut = CSV.read("escenarios_futuros_binarios.csv", DataFrame)
ret_fut_B = df_fut[:, 2]; ret_fut_A = df_fut[:, 3]

# ==============================================================================
# 2. CONFIGURACIÓN DEL ÁRBOL
# ==============================================================================
T = 6
cap_inicial = 2000.0
cap_objetivo = 4000.0 

probs_iniciales = [0.10, 0.20, 0.40, 0.20, 0.10] 
n_hojas = 5 * 2^(T-1) 

probs_hojas = zeros(n_hojas)
bloque_inicial = Int(n_hojas / 5)

for i in 1:5
    p_hoja = probs_iniciales[i] * (0.5)^(T-1)
    idx_inicio = (i-1)*bloque_inicial + 1
    idx_fin    = i*bloque_inicial
    probs_hojas[idx_inicio:idx_fin] .= p_hoja
end

println("Optimizando CVaR al 15% (Alta Seguridad)...")

# ==============================================================================
# 3. MODELO DE OPTIMIZACIÓN (ENFOQUE CVaR 15%)
# ==============================================================================

model = Model(Gurobi.Optimizer)
set_silent(model)

# Variables
@variable(model, x[1:length(activos), 1:T, 1:n_hojas] >= 0)
@variable(model, Wealth[1:T+1, 1:n_hojas] >= 0)

# --- VARIABLES PARA CVaR ---
alpha = 0.15  # 15% de riesgo aceptado
@variable(model, var_corte) 
@variable(model, shortfall[1:n_hojas] >= 0)

# --- RESTRICCIONES DEL ÁRBOL ---
@constraint(model, [n=1:n_hojas], Wealth[1, n] == cap_inicial)

for t in 1:T
    if t == 1; grupos = [1:n_hojas]; else
        n_grupos = 5 * 2^(t-2)
        tam_grupo = Int(n_hojas / n_grupos)
        grupos = [ (g-1)*tam_grupo+1 : g*tam_grupo for g in 1:n_grupos ]
    end

    for rango in grupos
        lider = first(rango)
        for n in rango
            if n != lider; @constraint(model, x[:, t, n] .== x[:, t, lider]); end
            @constraint(model, sum(x[:, t, n]) == Wealth[t, n])
            
            for i in 1:length(activos); @constraint(model, x[i, t, n] <= 0.40 * Wealth[t, n]); end
            @constraint(model, sum(x[i, t, n] for i in [5,6,7]) <= 0.20 * Wealth[t, n])
        end

        if t == 1
            bloque_hijos = Int(n_hojas / 5); lista_rets = [ret_MB, ret_B, ret_N, ret_A, ret_MA]
            for rama in 1:5
                inicio = (rama-1)*bloque_hijos + 1; fin = rama*bloque_hijos
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

# --- OBJETIVO CVaR ---
for n in 1:n_hojas
    @constraint(model, shortfall[n] >= var_corte - Wealth[T+1, n])
end

# MAXIMIZAR la media de la cola mala (CVaR).
# CORRECCIÓN AQUÍ: Se añade 'for n in 1:n_hojas'
@objective(model, Max, var_corte - (1/alpha) * sum(probs_hojas[n] * shortfall[n] for n in 1:n_hojas))

println("Ejecutando optimización...")
optimize!(model)

# ==============================================================================
# 4. RESULTADOS Y VISUALIZACIÓN
# ==============================================================================

if termination_status(model) == MOI.OPTIMAL
    println("\n" * "="^50)
    println(" PORTFOLIO CVaR 15% (Alta Seguridad)")
    println(" Se ignoran solo el 15% de los peores escenarios.")
    println("="^50)
    
    for i in 1:length(activos)
        dinero = value(x[i, 1, 1])
        peso = dinero / cap_inicial
        if peso > 0.001
            println(rpad(activos[i], 15), ": ", lpad(round(peso*100, digits=2), 5), "%")
        end
    end
    
    vals_finales = value.(Wealth[T+1, :])
    
    # Cálculo manual de métricas para mostrar
    df_risk = DataFrame(Wealth = vals_finales, Prob = probs_hojas)
    sort!(df_risk, :Wealth)
    df_risk.CumProb = cumsum(df_risk.Prob)
    
    idx_var15 = findfirst(x -> x >= 0.15, df_risk.CumProb)
    var_15_val = df_risk.Wealth[idx_var15]
    
    println("-"^50)
    println("Riqueza Esperada:   ", round(Int, sum(vals_finales .* probs_hojas)), " €")
    println("VaR 15% (El límite):", round(Int, var_15_val), " €")
    println("Peor Caso Absoluto: ", round(Int, minimum(vals_finales)), " €")
    println("-"^50)

    # --- HISTOGRAMA ---
    println("\nGenerando distribución...")
    
    p_hist = histogram(vals_finales, weights=probs_hojas,
        bins=50, label="Escenarios", color=:teal, linecolor=:white, alpha=0.6,
        title="Distribución con VaR 15%", xlabel="Riqueza Final (€)")
    
    vline!(p_hist, [var_15_val], label="Corte VaR 15%", color=:orange, linewidth=3)
    vline!(p_hist, [cap_inicial], label="Inicio (20k)", color=:red, linestyle=:dot, linewidth=2)
    
    display(p_hist)

else
    println("El modelo no encontró solución óptima: ", termination_status(model))
end