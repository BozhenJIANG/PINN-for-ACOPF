import PowerModels
import Ipopt
import Statistics
import Random
import Distributions
using DelimitedFiles
using Dates

function get_PQ_bus_set(PM_network_data)

    bus_set = []
    for (i, bus) in PM_network_data["bus"]
        if bus["bus_type"] == 1
            push!(bus_set, parse(Int64, i))
        end
    end

    bus_set = sort(unique(bus_set))
    return bus_set
end

function  _sort_according_index(_data,_list,_act_dim)
    sort_data = []
    for _i in range(1,_act_dim,_act_dim)
        _temp_index = 1
        for _j in _list
            if _i == _j 
                push!(sort_data,round.(_data[_temp_index],digits=8))
            else 
                _temp_index += 1
            end
        end
    end
    return Float64.(sort_data)    
    
end

##########################################################################################
##########################################################################################
##########################################################################################
const global balance_ref_id = "30"
const global gen_num = 54
const global bus_num = 118
network_data = PowerModels._parse_matpower_string(read("./case118.m", String))

load_pd_ori = []
load_qd_ori = []
for data in network_data["bus"]
    push!(load_pd_ori,[data["pd"],data["bus_i"]])
    push!(load_qd_ori,[data["qd"],data["bus_i"]])
end

load_pd_sort = sort!(load_pd_ori, by = x -> x[2])
load_qd_sort = sort!(load_qd_ori, by = x -> x[2])

X_con = [] 
X_in = []
X_other_information = []

PM_network_data = PowerModels._matpower_to_powermodels!(network_data)
PowerModels.correct_network_data!(PM_network_data)


for i in range(1,10000,10000)
    load_pd = [[row[1]+row[1]*Random.rand(Distributions.Normal(-0.1,0.03), 1)[1],row[2]] for row in load_pd_sort]
    load_qd = [[row[1]+row[1]*Random.rand(Distributions.Normal(-0.1,0.03), 1)[1],row[2]]  for row in load_qd_sort]

    load_length = length(load_pd)
    for (name, load) in PM_network_data["load"]
        for index in range(1,load_length,load_length)
            if load["source_id"][2] == load_pd[Int64(index)][2]
                load["pd"] = load_pd[Int64(index)][1]/100.0
                load["qd"] = load_qd[Int64(index)][1]/100.0
            end
        end
    end

    ACPP_model = PowerModels.instantiate_model(PM_network_data, PowerModels.ACPPowerModel, PowerModels.build_opf)
    opf_result = PowerModels.optimize_model!(ACPP_model, optimizer=Ipopt.Optimizer)
    
    PowerModels.update_data!(PM_network_data, opf_result["solution"])
   
    _temp_result = opf_result["termination_status"]
    
    if string(opf_result["termination_status"]) == "LOCALLY_SOLVED"
        PowerModels.update_data!(PM_network_data, opf_result["solution"])
       
        sort_index = []
        action = []
        _action = []
        # active power
        for (name,data) in opf_result["solution"]["gen"]
            #需要按name从大到小排序！！！
            if name == balance_ref_id # 机组是无功机组 
                push!(sort_index,parse(Int64,name))     
                push!(action,(data["pg"]-PM_network_data["gen"][name]["pmin"])/(PM_network_data["gen"][name]["pmax"]+0.001-PM_network_data["gen"][name]["pmin"]))
                push!(_action,data["pg"]*100)
            else
                push!(sort_index,parse(Int64,name)) 
                push!(action,(data["pg"]-PM_network_data["gen"][name]["pmin"])/(PM_network_data["gen"][name]["pmax"]-PM_network_data["gen"][name]["pmin"]))
                push!(_action,data["pg"]*100)
            end

        end
        
        for (name,data) in opf_result["solution"]["bus"]
            if PM_network_data["bus"][name]["bus_type"] == 2 || PM_network_data["bus"][name]["bus_type"] == 3
                push!(sort_index,parse(Int64,name)+Int64(gen_num))
                push!(action,data["vm"])
                push!(_action,data["vm"])
            end
        end   

        _action = _sort_according_index(_action, sort_index, gen_num+bus_num)
        action = _sort_according_index(action, sort_index, gen_num+bus_num)

        #  这里存的是原始值！

        push!(X_in,_action)
        push!(X_con,vcat([row[1] for row in load_pd],[row[1] for row in load_qd]))

        # PV节点的无功，PQ节点的电压，PQV节点的相角
        PV_Q = []
        for i in range(1,gen_num)
            for (name,data) in PM_network_data["gen"]
                if i == parse(Int64,name)
                    push!(PV_Q,data["qg"])
                end
            end
        end

        PQ_bus_set = get_PQ_bus_set(PM_network_data)
        PQ_V = []
        for i in PQ_bus_set
            for (name,data) in PM_network_data["bus"]
                if i == parse(Int64,name)
                    push!(PQ_V,data["vm"])
                end
            end
        end
    

        PQV_theta = []
        for i in range(1,bus_num)
            for (name,data) in PM_network_data["bus"]
                if i == parse(Int64,name)
                    push!(PQV_theta,data["va"])
                end
            end
        end
        push!(X_other_information,vcat(PV_Q,PQ_V,PQV_theta))
    else
        println(opf_result["termination_status"])
    end
end

open("./Dataset/X_con_118_train.txt", "w") do io
    writedlm(io, X_con) # 保存到文件
end

open("./Dataset/X_in_118_train.txt", "w") do io
    writedlm(io, X_in) # 保存到文件
end

open("./Dataset/X_other_information_118_train.txt", "w") do io
    writedlm(io, X_other_information) # 保存到文件
end


X_con = [] 
X_in = []
X_other_information = []

start_time = now()
for i in range(1,6000,6000)

    load_pd = [[row[1]+row[1]*Random.rand(Distributions.Normal(0.1,0.03), 1)[1],row[2]] for row in load_pd_sort]
    load_qd = [[row[1]+row[1]*Random.rand(Distributions.Normal(0.1,0.03), 1)[1],row[2]]  for row in load_qd_sort]


    load_length = length(load_pd)

    for (name, load) in PM_network_data["load"]
        for index in range(1,load_length,load_length)
            if load["source_id"][2] == load_pd[Int64(index)][2]
                load["pd"] = load_pd[Int64(index)][1]/100.0
                load["qd"] = load_qd[Int64(index)][1]/100.0
            end
        end
    end

    ACPP_model = PowerModels.instantiate_model(PM_network_data, PowerModels.ACPPowerModel, PowerModels.build_opf)
    opf_result = PowerModels.optimize_model!(ACPP_model, optimizer=Ipopt.Optimizer)

    _temp_result = opf_result["termination_status"]
    
    if string(opf_result["termination_status"]) == "LOCALLY_SOLVED"
        PowerModels.update_data!(PM_network_data, opf_result["solution"])
        sort_index = []
        action = []
        _action = []
        # active power
        for (name,data) in opf_result["solution"]["gen"]
            #需要按name从大到小排序！！！
            if name == balance_ref_id # 机组是无功机组 
                push!(sort_index,parse(Int64,name))     
                push!(action,(data["pg"]-PM_network_data["gen"][name]["pmin"])/(PM_network_data["gen"][name]["pmax"]+0.001-PM_network_data["gen"][name]["pmin"]))
                push!(_action,data["pg"]*100)
            else
                push!(sort_index,parse(Int64,name)) 
                push!(action,(data["pg"]-PM_network_data["gen"][name]["pmin"])/(PM_network_data["gen"][name]["pmax"]-PM_network_data["gen"][name]["pmin"]))
                push!(_action,data["pg"]*100)
            end

        end

        # bus vm
        for (name,data) in opf_result["solution"]["bus"]
            if PM_network_data["bus"][name]["bus_type"] == 2 || PM_network_data["bus"][name]["bus_type"] == 3
                push!(sort_index,parse(Int64,name)+Int64(gen_num))
                push!(action,data["vm"])
                push!(_action,data["vm"])
            end
        end   
        _action = _sort_according_index(_action, sort_index, gen_num+bus_num)
        action = _sort_according_index(action, sort_index, gen_num+bus_num)

        push!(X_in,_action)
        push!(X_con,vcat([row[1] for row in load_pd],[row[1] for row in load_qd]))

        # PV节点的无功，PQ节点的电压，PQV节点的相角
        PV_Q = []
        for i in range(1,gen_num)
            for (name,data) in PM_network_data["gen"]
                if i == parse(Int64,name)
                    push!(PV_Q,data["qg"])
                end
            end
        end

        PQ_bus_set = get_PQ_bus_set(PM_network_data)
        PQ_V = []
        for i in PQ_bus_set
            for (name,data) in PM_network_data["bus"]
                if i == parse(Int64,name)
                    push!(PQ_V,data["vm"])
                end
            end
        end
    
        PQV_theta = []
        for i in range(1,bus_num)
            for (name,data) in PM_network_data["bus"]
                if i == parse(Int64,name)
                    push!(PQV_theta,data["va"])
                end
            end
        end
        push!(X_other_information,vcat(PV_Q,PQ_V,PQV_theta))

    else
        println(opf_result["termination_status"])
    end
end
end_time = now()
time_difference = end_time - start_time
println("Time difference: ", time_difference)

open("./Dataset/X_con_118_test.txt", "w") do io
    writedlm(io, X_con) # 保存到文件
end

open("./Dataset/X_in_118_test.txt", "w") do io
    writedlm(io, X_in) # 保存到文件
end

open("./Dataset/X_other_information_118_test.txt", "w") do io
    writedlm(io, X_other_information) # 保存到文件
end

println("case118 finished !")


