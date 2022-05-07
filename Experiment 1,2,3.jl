
using DataFrames
using CSV
using Statistics
using DataStructures
using Distributions
using DelimitedFiles
using Random


#importing data

cd("C:/Users/79267/Documents/UW PhD/Spring 2022/ECON751 Labor economics/Part 2/Empirical Project")

data=readdlm("data_age4554.txt")
df = DataFrames.DataFrame(data, :auto)
df = df[2:30121,:]
df=rename!(df,[:id,:age,:lfp,:x,:wage,:educ,:lfp0,:hinc])


K=45 ##the maximum first period experience level, which is 24, +10 for now, K is length so add 1. Don't need to go any higher at this point
K_max=K-1
k_grid=Int.(collect(range(0, stop = K_max, length = K)))
A_final=64 ##A for terminating period
A_init=45
T_data=20 ##Periods
δ=0.95

N=1506
M=1 ##M is number of simulations


##setting parameters

α_1=-14500
α_2=-0.25
α_3=-100
α_4=-200
b=1000

β_1=9.93
β_2=0.01
β_3=-0.0002
β_4=0.035
σ_ξ=0.194

### tax rate
τ = 0

##functions to run##
function working_by_age_55_64(df_data)
    participation_by_age=zeros(10)
    for t=55:64
        df_age=filter(row -> row.age==t, df_data)
        participation_by_age[t-54]=mean(df_age.lfp)
    end
    return participation_by_age
end

#5 labor participation by lagged labor participation



function cutoffs_analytical(xi_matrix) ####This functionmcalculates cutoffs
    E_MAX=zeros(N,K,T_data)
    xi_star_matrix=zeros(N,K,T_data)  ##place to store cutoffs. This function use real data rather than grid so don't need grid for the other state variables
    for i=1:N
        println("this is", i, "person")
        for t=T_data:-1:1 #from backwards
            row=Matrix(filter(row -> row.id ==i && row.age==44+t, df)) #select i's info in time t. e.g.if T_data is 10, ag =44+10=54, which is the last period in data
            y=row[8] #the last row is husband's income
            s=row[6] #the sixth row is education
            if t==T_data #for last period
                for k=0:K_max #last period possible experience 0 to k_max of 34, k is experience so start from 0
                    log_inside= -α_1 - α_2*(1-τ)*y + b*(1+α_2) - α_3*k -α_4*s #dealing with the domain error problem
                    if log_inside >= 0
                        xi_star=log(-α_1 - α_2*(1-τ)*y + b*(1+α_2) - α_3*k -α_4*s) - log(1 +α_2)- (β_1 + β_2*k + β_3*k^2 + β_4*s)-log(1-τ)  #xi_star us calculated for any state variable k experience in the last year
                        xi_star_matrix[i,k+1,t]=xi_star #to save need to add one to k to get index.
                    elseif log_inside <0
                        xi_star=-1000000000000
                        xi_star_matrix[i,k+1,t]=xi_star
                    end
                    X_A= exp(β_1 + β_2*k + β_3*k^2 + β_4*s)
                    E_MAX[i,k+1,t]=(α_1 + (1+α_2)*((1-τ)*y-b) + α_3*k + α_4*s)*(1-cdf.(Normal(),xi_star/σ_ξ)) + (1+α_2)*(1-τ)*X_A*exp(0.5*σ_ξ^2)*(1-cdf.(Normal(),(xi_star-σ_ξ^2)/σ_ξ)) + (1-τ)*y*cdf.(Normal(),xi_star/σ_ξ)
                end
            elseif t<=T_data-1 #if not last period
                for k=0:K_max-(T_data-t) #e.g. if this period is t=6, the max experience possible is 34-(10-6)=30
                    log_inside = -α_1 - α_2*(1-τ)*y + b*(1+α_2) - α_3*k -α_4*s + δ*(E_MAX[i,k+1,t+1]-E_MAX[i,k+2,t+1])
                    if log_inside >= 0 #dealing with the domain error problem
                        xi_star=log(-α_1 - α_2*(1-τ)*y + b*(1+α_2) - α_3*k -α_4*s + δ*(E_MAX[i,k+1,t+1]-E_MAX[i,k+2,t+1])) - log(1+α_2)-(β_1 + β_2*k + β_3*k^2 + β_4*s) -log(1-τ) #T-1 period cutoff only depends on expected for T period
                        xi_star_matrix[i,k+1,t]=xi_star
                    elseif log_inside <0
                        xi_star=-1000000000000
                        xi_star_matrix[i,k+1,t]=xi_star
                    end
                    X_A= exp(β_1 + β_2*k + β_3*k^2 + β_4*s)
                    E_MAX[i,k+1,t]=(α_1 + (1+α_2)*((1-τ)*y-b) + α_3*k + α_4*s + E_MAX[i,k+1+1,t+1])*(1-cdf.(Normal(),xi_star/σ_ξ)) + (1+α_2)*(1-τ)*X_A*exp(0.5*σ_ξ^2)*(1-cdf.(Normal(),(xi_star-σ_ξ^2)/σ_ξ))
                    + ((1-τ)*y + E_MAX[i,k+1,t+1])*cdf.(Normal(),xi_star/σ_ξ)
                end
            end
        end
    end
    return xi_star_matrix
end



function getendo_new(xi_star_matrix, xi_matrix) ##this generates endogenous x and k for each simulation using initial experience
    k_vector=zeros(N,T_data,M) #experience of each person at each period in each simulation
    x=zeros(N,T_data,M) #action of each person at each period in each simulation
    for m=1:M
        for i=1:N
            println("this is", i, "person")
            row=Matrix(filter(row -> row.id ==i && row.age==45, df)) #only need the starting experience
            for t=1:T_data
                xi=xi_matrix[i,t,m] #get shock from shock matrix
                if t==1
                    k=row[4]
                    k_vector[i,t,m]=k #experience is 4th column, need to make sure this is an integer
                    xi_cutoff=xi_star_matrix[i,k+1,t] #add 1 for matrix indice
                    if xi >= xi_cutoff
                        x[i,t,m]=1 #t
                        k_vector[i,t+1,m]=k+1 #update experience next period
                    elseif xi < xi_cutoff
                        k_vector[i,t+1,m]=k
                        x[i,t,m]=0
                    end
                elseif 1<t<T_data
                    k=floor(Int,k_vector[i,t,m]) ##I don't know why I need to convert as interger again but if I don't do it I get error.
                    xi_cutoff=xi_star_matrix[i,k+1,t]
                    if xi >= xi_cutoff
                        k_vector[i,t+1,m]=k+1 #update experience next period
                        x[i,t,m]=1 #this period participation is 1
                    elseif xi < xi_cutoff
                        k_vector[i,t+1,m]=k
                        x[i,t,m]=0
                    end
                else t==T_data #do not need to update next period.
                    k=floor(Int,k_vector[i,t,m])
                    xi_cutoff=xi_star_matrix[i,k+1,t]
                    if xi >= xi_cutoff
                        x[i,t,m]=1 #this period participation is 1
                    elseif xi < xi_cutoff
                        x[i,t,m]=0
                    end
                end
            end
        end
    end
    return x, k_vector
end

function simulate_data(lfp_vector,k_vector,wages_vector,df_for_calib) ##create function to store all simulated data in a datafram similar to the real data
    s_id=zeros(N*T_data) #create vectors store information, the number of rows is N times time period
    s_age=zeros(N*T_data)
    s_lfp=zeros(N*T_data)
    s_x=zeros(N*T_data)
    s_wages=zeros(N*T_data)
    s_educ=zeros(N*T_data)
    sim_data=zeros(N*T_data,7,M) ##to store the six things we care about, no initial labor or husband's wage
    for m=1:M
        s_id=df_for_calib.id #s_id is just id information from the df
        s_age=df_for_calib.age
        s_lfp=vec(lfp_vector[:,:,m]') #need ' to go by id first go through all t
        s_x=vec(k_vector[:,:,m]')
        s_wages=vec(wages_vector[:,:,m]')
        s_educ=df_for_calib.educ
        s_hinc=df_for_calib.hinc
        sim=DataFrames.DataFrame(hcat(s_id, s_age,s_lfp, s_x,s_wages,s_educ,s_hinc), :auto) #convert to dataframe format
        sim_data[:,:,m].=rename!(sim,[:id,:age,:lfp,:x,:wage,:educ,:hinc]) #rename data
    end
    return sim_data
end


### Experiment 1
function get_results_experiment1() #this computes results for experiment 1
    Random.seed!(1234); #generating simulated results
    xi_matrix = reshape(rand(Normal(0, σ_ξ), N*T_data*M), N,T_data,M) #draw shocks
    xi_star_matrix= cutoffs_analytical(xi_matrix) #compute cutoffs
    lfp_vector, k_vector = getendo_new(xi_star_matrix, xi_matrix) #this calculates participation and experience
    educ_change=reshape(df.educ,T_data,N) #need to get pick up education
    educ=educ_change' #do this to get in the right form
    wages_vector = exp.(β_1.+ β_2.*k_vector.+ β_3.*k_vector.^2 .+  β_4*educ.+xi_matrix)
    sim_data_all=simulate_data(lfp_vector,k_vector,wages_vector,df) #convert the results for all the simulations into a datraframe, this is in N*T, 6, m dimension

    working_by_educ=zeros(5,M) ##setting up matrix for storing the moment differences
    working_by_age=zeros(10,M)

    for m=1:M
        sim_data=DataFrames.DataFrame(sim_data_all[:,:,m], :auto) #take out one simulation's data
        sim_data=rename!(sim_data,[:id,:age,:lfp,:x,:wage,:educ])
        sim_data_55_64=filter(row -> row.age>=55, sim_data) ##only calculate for the out of sample group

        sim_data_educ_under_12=filter(row -> row.educ<=11, sim_data_55_64) ##only calculate participation for the out of sample group
        sim_data_educ_12=filter(row -> row.educ==12, sim_data_55_64)
        sim_data_educ_13_15=filter(row -> 13<=row.educ<=15,sim_data_55_64)
        sim_data_educ_16_plus=filter(row -> row.educ>=16, sim_data_55_64)

        working_by_educ[1,m]=mean(sim_data_55_64.lfp) #for each simulation compare results and store
        working_by_educ[2,m]=mean(sim_data_educ_under_12.lfp)
        working_by_educ[3,m]=mean(sim_data_educ_12.lfp)
        working_by_educ[4,m]=mean(sim_data_educ_13_15.lfp)
        working_by_educ[5,m]=mean(sim_data_educ_16_plus.lfp)
        working_by_age[:,m]=working_by_age_55_64(sim_data_55_64)
    end
        working_by_educ_mean=mean(working_by_educ,dims=2) #average over the second dimension, which is M
        working_by_age_mean=mean(working_by_age,dims=2)

        return working_by_educ_mean, working_by_age_mean
end

@elapsed working_by_educ_extrapolate, working_by_age_extrapolate=get_results_experiment1()
@show working_by_educ_extrapolate, working_by_age_extrapolate

### Experiment 2

### Took the same code as for experiment 1 and changed a bit
### need to find only average number of years worked from 45 to 54 and from 55 to 64.
function get_results_experiment2() #this computes results for experiment 1
    Random.seed!(1234); #generating simulated results
    xi_matrix = reshape(rand(Normal(0, σ_ξ), N*T_data*M), N,T_data,M) #draw shocks
    xi_star_matrix= cutoffs_analytical(xi_matrix) #compute cutoffs
    lfp_vector, k_vector = getendo_new(xi_star_matrix, xi_matrix) #this calculates participation and experience
    educ_change=reshape(df.educ,T_data,N) #need to get pick up education
    educ=educ_change' #do this to get in the right form
    wages_vector = exp.(β_1.+ β_2.*k_vector.+ β_3.*k_vector.^2 .+  β_4*educ.+xi_matrix)
    sim_data_all=simulate_data(lfp_vector, k_vector, wages_vector, df) #convert the results for all the simulations into a datraframe, this is in N*T, 6, m dimension

    average_working_45_54=zeros(1,M) ##setting up matrix for storing the moment differences
    average_working_55_64=zeros(1,M)
    tax_revenues = zeros(1, M)

    for m=1:M
        sim_data=DataFrames.DataFrame(sim_data_all[:,:,m], :auto) #take out one simulation's data
        sim_data=rename!(sim_data,[:id,:age,:lfp,:x,:wage,:educ,:hinc])
        sim_data_45_54=filter(row -> 45<=row.age<=54, sim_data) ##only calculate for the out of sample group
        sim_data_55_64=filter(row -> 55<=row.age<=64, sim_data) ##only calculate for the out of sample group

        average_working_45_54[1, m] = mean(sim_data_45_54.lfp)
        average_working_55_64[1, m] = mean(sim_data_55_64.lfp)

        ### To compute tax_revenues
        sim_data_working = filter(row -> row.lfp == 1, sim_data)

        tax_revenues[1, m] = sum(τ.*(sim_data_working.wage .+ sim_data_working.hinc))
    end
        average_working_45_54_mean=mean(average_working_45_54,dims = 2) #average over the second dimension, which is M
        average_working_55_64_mean=mean(average_working_55_64,dims = 2)
        tax_revenues_mean = mean(tax_revenues, dims = 2)

        return average_working_45_54_mean, average_working_55_64_mean, tax_revenues_mean
end

τ=0.0
@elapsed working_45_54_mean, working_55_64_mean, tax_revenues_mean = get_results_experiment2()
@show working_45_54_mean, working_55_64_mean, tax_revenues_mean

τ=0.1
@elapsed working_45_54_mean_exp2, working_55_64_mean_exp2, tax_revenues_mean_exp2 = get_results_experiment2()
@show working_45_54_mean_exp2, working_55_64_mean_exp2, tax_revenues_mean_exp2


### τ=0, working_45_54_mean = 0.536, working_55_64_mean = 0.494
### Note: the result for age group 55-64 is consistent with what was obtained in the experiment 1, working_by_educ_extrapolate[1]

### τ=0.1, working_45_54_mean = 0.389, working_55_64_mean = 0.378, tax revenues = 7.23*e7

### Experiment 3

### Took the same code as for experiment 2 and instead of calling cutoffs_analytical(), I call cutoffs_analytical_exp3()


function cutoffs_analytical_exp3(xi_matrix) ####This functionmcalculates cutoffs
    E_MAX=zeros(N,K,T_data)
    xi_star_matrix=zeros(N,K,T_data)  ##place to store cutoffs. This function use real data rather than grid so don't need grid for the other state variables
    #τ1 = 0.10
    #τ2 = 0.50
    I = 50000
    for i=1:N
        println("this is", i, "person")
        for t=T_data:-1:1 #from backwards
            row=Matrix(filter(row -> row.id ==i && row.age==44+t, df)) #select i's info in time t. e.g.if T_data is 10, ag =44+10=54, which is the last period in data
            y=row[8] #the last row is husband's income
            s=row[6] #the sixth row is education
            if t==T_data #for last period
                for k=0:K_max #last period possible experience 0 to k_max of 34, k is experience so start from 0
                    if y > I ### use two different tax rates
                        τ1 = 0.1
                        τ2 = 0.2
                        log_inside= -α_1 - α_2*(1-τ2)*(y-I) -α_2*(1-τ1)*I + b*(1+α_2) - α_3*k -α_4*s #dealing with the domain error problem
                        if log_inside >= 0
                            xi_star=log(-α_1 - α_2*(1-τ2)*(y-I) -α_2*(1-τ1)*I + b*(1+α_2) - α_3*k -α_4*s) - log(1 +α_2)- (β_1 + β_2*k + β_3*k^2 + β_4*s)-log(1-τ2)  #xi_star us calculated for any state variable k experience in the last year
                            xi_star_matrix[i,k+1,t]=xi_star #to save need to add one to k to get index.
                        elseif log_inside <0
                            xi_star=-1000000000000
                            xi_star_matrix[i,k+1,t]=xi_star
                        end
                        X_A= exp(β_1 + β_2*k + β_3*k^2 + β_4*s)
                        E_MAX[i,k+1,t]=(α_1 + (1+α_2)*((1-τ2)*(y-I)+(1-τ1)*I-b) + α_3*k + α_4*s)*(1-cdf.(Normal(),xi_star/σ_ξ)) +
                                        (1+α_2)*(1-τ2)*X_A*exp(0.5*σ_ξ^2)*(1-cdf.(Normal(),(xi_star-σ_ξ^2)/σ_ξ)) + (1-τ2)*((1-τ2)*(y-I)+(1-τ1)*I)*cdf.(Normal(),xi_star/σ_ξ)
                    end
                    if y <= I
                        wage_eps = I-y
                        X_A= exp(β_1 + β_2*k + β_3*k^2 + β_4*s)
                        τ1 = 0.1
                        τ2 = 0.1
                        V1 = α_1 + (1+α_2)*((1-τ2)*(y-I)+(1-τ1)*I-b) + α_3*k + α_4*s + (1+α_2)*(1-τ2)*X_A
                        V0 = (1-τ2)*(y-I)+(1-τ1)*I
                        if V1 < V0 ### need to compute cutoff using two different tax rates
                            τ2 =0.2
                        end
                        log_inside= -α_1 - α_2*(1-τ2)*(y-I) -α_2*(1-τ1)*I + b*(1+α_2) - α_3*k -α_4*s #dealing with the domain error problem
                        if log_inside >= 0
                            xi_star=log(-α_1 - α_2*(1-τ2)*(y-I) -α_2*(1-τ1)*I + b*(1+α_2) - α_3*k -α_4*s) - log(1 +α_2)- (β_1 + β_2*k + β_3*k^2 + β_4*s)-log(1-τ2)  #xi_star us calculated for any state variable k experience in the last year
                            xi_star_matrix[i,k+1,t]=xi_star #to save need to add one to k to get index.
                        elseif log_inside <0
                            xi_star=-1000000000000
                            xi_star_matrix[i,k+1,t]=xi_star
                        end
                        E_MAX[i,k+1,t]=(α_1 + (1+α_2)*((1-τ2)*(y-I)+(1-τ1)*I-b) + α_3*k + α_4*s)*(1-cdf.(Normal(),xi_star/σ_ξ)) +
                                        (1+α_2)*(1-τ2)*X_A*exp(0.5*σ_ξ^2)*(1-cdf.(Normal(),(xi_star-σ_ξ^2)/σ_ξ)) + (1-τ2)*((1-τ2)*(y-I)+(1-τ1)*I)*cdf.(Normal(),xi_star/σ_ξ)
                    end
                end
            elseif t<=T_data-1 #if not last period
                for k=0:K_max-(T_data-t) #e.g. if this period is t=6, the max experience possible is 34-(10-6)=30
                    if y > I
                        τ1 = 0.1
                        τ2 = 0.2
                        log_inside = -α_1 - α_2*(1-τ2)*(y-I) -α_2*(1-τ1)*I + b*(1+α_2) - α_3*k -α_4*s + δ*(E_MAX[i,k+1,t+1]-E_MAX[i,k+2,t+1])
                        if log_inside >= 0 #dealing with the domain error problem
                            xi_star=log(-α_1 - α_2*(1-τ2)*(y-I) -α_2*(1-τ1)*I + b*(1+α_2) - α_3*k -α_4*s + δ*(E_MAX[i,k+1,t+1]-E_MAX[i,k+2,t+1])) - log(1+α_2)-
                                    (β_1 + β_2*k + β_3*k^2 + β_4*s) -log(1-τ2) #T-1 period cutoff only depends on expected for T period
                            xi_star_matrix[i,k+1,t]=xi_star
                        elseif log_inside <0
                            xi_star=-1000000000000
                            xi_star_matrix[i,k+1,t]=xi_star
                        end
                        X_A= exp(β_1 + β_2*k + β_3*k^2 + β_4*s)
                        E_MAX[i,k+1,t]=(α_1 + (1+α_2)*((1-τ2)*(y-I)+(1-τ1)*I-b) + α_3*k + α_4*s + E_MAX[i,k+1+1,t+1])*(1-cdf.(Normal(),xi_star/σ_ξ)) +
                                        (1+α_2)*(1-τ2)*X_A*exp(0.5*σ_ξ^2)*(1-cdf.(Normal(),(xi_star-σ_ξ^2)/σ_ξ))
                                        + ((1-τ2)*(y-I)+(1-τ1)*I + E_MAX[i,k+1,t+1])*cdf.(Normal(),xi_star/σ_ξ)
                    end
                    if y <= I
                        wage_eps = I-y
                        X_A= exp(β_1 + β_2*k + β_3*k^2 + β_4*s)
                        τ1 = 0.1
                        τ2 = 0.1
                        V1 = α_1 + (1+α_2)*((1-τ2)*(y-I)+(1-τ1)*I-b) + α_3*k + α_4*s + (1+α_2)*(1-τ2)*X_A + δ*E_MAX[i,k+1+1,t+1]
                        V0 = (1-τ2)*(y-I)+(1-τ1)*I + δ*E_MAX[i,k+1,t+1]
                        if V1 < V0
                            τ2 = 0.2
                        end
                        log_inside = -α_1 - α_2*(1-τ2)*(y-I) -α_2*(1-τ1)*I + b*(1+α_2) - α_3*k -α_4*s + δ*(E_MAX[i,k+1,t+1]-E_MAX[i,k+2,t+1])
                        if log_inside >= 0 #dealing with the domain error problem
                            xi_star=log(-α_1 - α_2*(1-τ2)*(y-I) -α_2*(1-τ1)*I + b*(1+α_2) - α_3*k -α_4*s + δ*(E_MAX[i,k+1,t+1]-E_MAX[i,k+2,t+1])) - log(1+α_2)-
                                    (β_1 + β_2*k + β_3*k^2 + β_4*s) -log(1-τ2) #T-1 period cutoff only depends on expected for T period
                            xi_star_matrix[i,k+1,t]=xi_star
                        elseif log_inside <0
                            xi_star=-1000000000000
                            xi_star_matrix[i,k+1,t]=xi_star
                        end
                        E_MAX[i,k+1,t]=(α_1 + (1+α_2)*((1-τ2)*(y-I)+(1-τ1)*I-b) + α_3*k + α_4*s + E_MAX[i,k+1+1,t+1])*(1-cdf.(Normal(),xi_star/σ_ξ)) +
                                        (1+α_2)*(1-τ2)*X_A*exp(0.5*σ_ξ^2)*(1-cdf.(Normal(),(xi_star-σ_ξ^2)/σ_ξ))
                                        + ((1-τ2)*(y-I)+(1-τ1)*I + E_MAX[i,k+1,t+1])*cdf.(Normal(),xi_star/σ_ξ)
                    end
                end
            end
        end
    end
    return xi_star_matrix
end


function get_results_experiment3() #this computes results for experiment 1
    Random.seed!(1234); #generating simulated results
    xi_matrix = reshape(rand(Normal(0, σ_ξ), N*T_data*M), N,T_data,M) #draw shocks
    xi_star_matrix= cutoffs_analytical_exp3(xi_matrix) #compute cutoffs
    lfp_vector, k_vector = getendo_new(xi_star_matrix, xi_matrix) #this calculates participation and experience
    educ_change=reshape(df.educ,T_data,N) #need to get pick up education
    educ=educ_change' #do this to get in the right form
    wages_vector = exp.(β_1.+ β_2.*k_vector.+ β_3.*k_vector.^2 .+  β_4*educ.+xi_matrix)
    sim_data_all=simulate_data(lfp_vector, k_vector, wages_vector, df) #convert the results for all the simulations into a datraframe, this is in N*T, 6, m dimension

    average_working_45_54=zeros(1,M) ##setting up matrix for storing the moment differences
    average_working_55_64=zeros(1,M)
    tax_revenues = zeros(1, M)

    τ1 = 0.1
    τ2 = 0.2
    I = 50000

    for m=1:M
        sim_data=DataFrames.DataFrame(sim_data_all[:,:,m], :auto) #take out one simulation's data
        sim_data=rename!(sim_data,[:id,:age,:lfp,:x,:wage,:educ,:hinc])
        sim_data_45_54=filter(row -> 45<=row.age<=54, sim_data) ##only calculate for the out of sample group
        sim_data_55_64=filter(row -> 55<=row.age<=64, sim_data) ##only calculate for the out of sample group

        average_working_45_54[1, m] = mean(sim_data_45_54.lfp)
        average_working_55_64[1, m] = mean(sim_data_55_64.lfp)

        ### To compute tax_revenues
        sim_data_working = filter(row -> row.lfp == 1, sim_data)

        tax_revenues[1, m] = sum(τ1.*(sim_data_working.wage .+ sim_data_working.hinc).*(sim_data_working.wage .+ sim_data_working.hinc .<= I) .+
                                (τ2.*(sim_data_working.wage .+ sim_data_working.hinc .- I).+ τ1.*I).*(sim_data_working.wage .+ sim_data_working.hinc .> I))
    end
        average_working_45_54_mean=mean(average_working_45_54,dims = 2) #average over the second dimension, which is M
        average_working_55_64_mean=mean(average_working_55_64,dims = 2)
        tax_revenues_mean = mean(tax_revenues, dims = 2)

        return average_working_45_54_mean, average_working_55_64_mean, tax_revenues_mean
end

@elapsed working_45_54_mean_exp3, working_55_64_mean_exp3, tax_revenues_mean_exp3 = get_results_experiment3()
@show working_45_54_mean_exp3, working_55_64_mean_exp3, tax_revenues_mean_exp3

### income > 50,000, then τ1 = 0.1, τ2 = 0.2,
### income <= 50,000, then τ1 =  τ2 = 0.1,
### working_45_54_mean = 0.371, working_55_64_mean = 0.284, tax revenues = 6.86*e7
