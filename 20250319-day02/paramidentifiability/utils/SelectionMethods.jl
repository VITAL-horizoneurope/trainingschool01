abstract type SelectionMethod end

struct TopKSelection <: SelectionMethod
    crit::Function
    k::Int
end

struct TopPercentSelection <: SelectionMethod
    crit::Function
    percent::Float64
end

struct SelectAll <: SelectionMethod end

function select_indices(criteria, method::TopKSelection)
    return partialsortperm(criteria, 1:min(method.k, length(criteria)))
end

function select_indices(criteria, method::TopPercentSelection)
    k = Int(round(method.percent * length(criteria)))
    return partialsortperm(criteria, 1:min(k, length(criteria)))
end

function select_indices(criteria, ::SelectAll)
    return 1:length(criteria)
end

function evaluate_and_select(parameters, ::SelectAll, args...)
    return parameters
end

function evaluate_and_select(parameters, method::SelectionMethod, args...)

    criteria = map(p -> method.crit(p, args...), eachrow(parameters))
    indices = select_indices(criteria, method)

    if isnan(criteria[1]) || isinf(criteria[1])
        println("Warning:\t top criterion value == $(criteria[1])")
    end

    return parameters[indices, :]
end






