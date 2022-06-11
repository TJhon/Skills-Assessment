---
title: "Julia"
---


### Data

```julia
using CSV, GLM, Plots
using DataFrames, Lathe, MLBase

using Lathe.preprocess: TrainTestSplit



wage = CSV.read("data/wage2015_subsample_inference.csv", DataFrame)

long_wag = stack(wage, [:shs, :hsg, :scl, :clg, :ad], [:rownames, :wage, :lwage, :sex, :mw, :so, :we, :ne, :exp1, :exp2, :exp3, :exp4, :occ, :occ2, :ind, :ind2])

filter!(n-> n.value >0, long_wag)
```

```
5150×18 DataFrame
  Row │ rownames  wage      lwage    sex      mw       so       we       ne
    ⋯
      │ Int64     Float64   Float64  Float64  Float64  Float64  Float64  Fl
oat ⋯
──────┼────────────────────────────────────────────────────────────────────
─────
    1 │      500   8.54701  2.14558      0.0      0.0      0.0      0.0    
  1 ⋯
    2 │      540  10.4396   2.3456       0.0      0.0      0.0      0.0    
  1
    3 │      691  14.9038   2.70162      0.0      0.0      0.0      0.0    
  1
    4 │      843   9.61538  2.26336      0.0      0.0      0.0      0.0    
  1
    5 │     1775   9.44231  2.2452       1.0      0.0      0.0      0.0    
  1 ⋯
    6 │     2523  24.0385   3.17966      0.0      0.0      0.0      0.0    
  1
    7 │     2614   8.24176  2.10921      0.0      0.0      0.0      0.0    
  1
    8 │     2633  15.3846   2.73337      0.0      0.0      0.0      0.0    
  1
  ⋮   │    ⋮         ⋮         ⋮        ⋮        ⋮        ⋮        ⋮       
 ⋮  ⋱
 5144 │    32537  13.986    2.63806      1.0      0.0      0.0      1.0    
  0 ⋯
 5145 │    32538  15.8654   2.76414      0.0      0.0      0.0      1.0    
  0
 5146 │    32548  24.4755   3.19767      1.0      0.0      0.0      1.0    
  0
 5147 │    32565  35.2564   3.56265      0.0      0.0      0.0      1.0    
  0
 5148 │    32591  19.7115   2.9812       1.0      0.0      0.0      1.0    
  0 ⋯
 5149 │    32626  38.4615   3.64966      0.0      0.0      0.0      1.0    
  0
 5150 │    32643  17.3077   2.85115      0.0      0.0      0.0      1.0    
  0
                                                11 columns and 5135 rows om
itted
```





### Modelos

```julia
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
```

```
4×4 DataFrame
 Row │ Model         type    r2        mse
     │ String        String  Float64   Float64
─────┼──────────────────────────────────────────
   1 │ Basic OLS     train   0.180664  0.270149
   2 │ Basic OLS     test    0.188305  0.26481
   3 │ Flexible OLS  train   0.213052  0.25947
   4 │ Flexible OLS  test    0.2228    0.266027
```





### Coef Plot

```julia
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
```

![](figures/ra_sa_jl_3_1.png)



### Wage - Experience

```julia
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
```

![](figures/ra_sa_jl_4_1.png)
