

using CSV, GLM, Plots
using DataFrames, Lathe, MLBase

using Lathe.preprocess: TrainTestSplit



wage = CSV.read("data/wage2015_subsample_inference.csv", DataFrame)

long_wag = stack(wage, [:shs, :hsg, :scl, :clg, :ad], [:rownames, :wage, :lwage, :sex, :mw, :so, :we, :ne, :exp1, :exp2, :exp3, :exp4, :occ, :occ2, :ind, :ind2])

filter!(n-> n.value >0, long_wag)

### Plots



### Modelos

tr_wg, ts_wg = TrainTestSplit(wage, .60)

ols_reg = lm(@formula(lwage ~ sex + exp1 + shs + hsg + mw + so + we + occ2 + ind2), tr_wg)
flx_reg = lm(@formula(lwage ~ sex+(exp1+exp2+exp3+exp4+shs+hsg +occ2+ind2 + mw + so + we)*(exp1+exp2+exp3+exp4+shs+hsg +occ2+ind2 + mw + so + we)), tr_wg)


y_real_train = tr_wg.lwage
y_real_test = ts_wg.lwage
y_fit_ols_train = predict(ols_reg, tr_wg)
y_fit_ols_test = predict(ols_reg, ts_wg)

y_fit_flx_train = predict(flx_reg, tr_wg)
y_fit_flx_test = predict(flx_reg, ts_wg)


r2_md = [r2(ols_reg), r2(flx_reg)]


function r_2_2(real, fit)
    a = sum((fit .- mean(real)).^2)
    b = sum((real .- mean(real)).^2)
    c = a/b
    return c    
end 

function mse(real, fit)
    a = (real .- fit).^2
    b = mean(a)
    return b
end

tbl = DataFrame(
    Model = ["Basic OLS", "Basic OLS", "Flexible OLS", "Flexible OLS"],
    type = ["train", "test", "train", "test"],
    r2 = 
    [r_2_2(y_real_train, y_fit_ols_train),
    r_2_2(y_real_test, y_fit_ols_test), 
    r_2_2(y_real_train, y_fit_flx_train),
    r_2_2(y_real_test, y_fit_flx_test)],
    mse = [
        mse(y_fit_ols_train, y_real_train),
        mse(y_fit_ols_test, y_real_test),
        mse(y_fit_flx_train, y_real_train),
        mse(y_fit_flx_test, y_real_test),
    ]
)


function coef_plot(m1, m2)
    n1 = ["Basic OLS", "Flexible OLS"]
    c1 = [coef(m1)[2], coef(m2)[2]]
    err1 = [GLM.stderror(m1)[2], GLM.stderror(m2)[2]]

    df = DataFrame(a = n1, b = c1, err = err1)
    scatter(
        df.a, df.b,
        legend = false,
        yerror = 1.96.*df.err,
        title = "Coefficient Plots"
        )
end

coef_plot(ols_reg, flx_reg)

long_wag


coll = filter(x -> x.variable == "clg", long_wag)
sort!(coll, :exp1)
hg = filter(x -> x.variable == "hsg", long_wag)
sort!(hg, :exp1)

col_md = lm(@formula(lwage ~ exp1 + exp1^2), coll)
hg_md = lm(@formula(lwage ~ exp1 + exp1^2), hg)
plot(coll.exp1, predict(col_md), label = "College")
plot!(hg.exp1, predict(hg_md), label = "High School") 
xlabel!("Years of Potential Experience")
ylabel!("Log Wage")
