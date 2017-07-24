function fast_logistf_fit(x::Array{Float64, 2}, y::Array{Int64, 1})
    const n = size(x, 1) 
    const k = size(x, 2) 
    const init = zeros(Float64, k)
    const col_fit = 1:k
    const offset = zeros(Float64, n)
    const weight = ones(n)
    const maxit = 50
    const maxhs = 15
    const maxstep = 15
    const lconv = 1e-05
    const gconv = 1e-05
    const xconv = 1e-05
    beta = init
    l_change = 5
    iter = 0
    pi = 1.0 ./ (1 + exp(-x * beta - offset))
    loglik = sum(weight[y .== 1] .* log(pi[y .== 1])) +
             sum(weight[y .== 0] .* log(1 - pi[y .== 0]))::Float64

    XW2 = x' * diagm((weight .* pi .* (1 - pi)) .^ 0.5)::Array{Float64, 2}
    Fisher = XW2 * XW2'::Array{Float64, 2}
    loglik = loglik + 0.5 * logdet(Fisher)::Float64

    evals = 1
    XX_covs = zeros(k, k)::Array{Float64, 2}
    h = Array{Float64, 1}
    while iter < maxit
        loglik_old = loglik::Float64
        beta_old = beta::Array{Float64, 1}
        XW2 = x' * diagm((weight .* pi .* (1 - pi)) .^ 0.5)::Array{Float64, 2}
        Q, R = qr(XW2')
        h = (Q .* Q) * ones(size(Q, 2))::Array{Float64, 1}
        U_star = x' * (weight .* (y - pi) + h .* (0.5 - pi))

        if col_fit[1] != 0
            XX_XW2 = x[:, col_fit]' * diagm((weight .* pi .* (1 - pi)) .^ 0.5)::Array{Float64, 2}
            XX_Fisher = XX_XW2 * XX_XW2'::Array{Float64, 2}::Array{Float64, 2}
            XX_covs[col_fit, col_fit] = inv(XX_Fisher)::Array{Float64, 2}
        end

        delta = XX_covs * U_star::Array{Float64, 1}
        mx = maximum(abs(delta)) / maxstep
        if mx > 1
            delta = delta / mx
        end

        if maxit > 0
            iter = iter + 1::Int64
            beta = beta + delta
            for halfs in 1:maxhs
                pi = 1.0 ./ (1 + exp(-x * beta - offset))
                loglik = sum(weight[y .== 1] .* log(pi[y .== 1])) +
                         sum(weight[y .== 0] .* log(1 - pi[y .== 0]))::Float64
                XW2 = x' * diagm((weight .* pi .* (1 - pi)) .^ 0.5)::Array{Float64, 2}
                Fisher = XW2 * XW2'::Array{Float64, 2}
                loglik = loglik + 0.5 * logdet(Fisher)::Float64
                evals = evals + 1
                l_change = loglik - loglik_old
                if loglik > loglik_old
                    break
                end
                beta = beta - delta .* 2.0 .^ (-float(halfs))
                
            end # end of for loop
        end # end of if statement
        flag_one = maximum(abs(delta)) .< xconv
        flag_two = all(abs(U_star[col_fit]) .< gconv) 
        flag_three = all(l_change .< lconv)

        if flag_one && flag_two && flag_three
            break
        end 
    end # end of while loop

    return beta, XX_covs, pi, h, loglik, iter, evals
end
