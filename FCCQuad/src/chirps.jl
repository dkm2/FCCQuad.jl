module Chirps

using HDF5, Fct
thresholds = [
  5   1.5
  6   4.0
  7   5.9
  8   7.3
  9   8.6
 10   9.7
 11  10.8
 12  11.8
 13  12.8
 14  13.8
 15  14.8
 16  15.8
 17  16.9
]

L=size(thresholds)[1]

offsets = accumulate(+,vcat([1],[1+2^(Int64(thresholds[i,1])-1) for i in 1:L]))

#even-degree Chebyshev coefficients of linear chirps
chirps = Vector{Complex{Float64}}(undef,offsets[end])

degrees = [2 ^ Int64(p) for p in thresholds[:,1]]
rates = 2.0 .^ thresholds[:,2]

function getchirp(i)# 1 <= i <= L
    view(chirps,offsets[i]:offsets[i+1]-1)
end

function store(file)
    h=h5open(file,"w") #overwrite
    for i in 1:L
        N = degrees[i]
        a = rates[i]
        s = Fct.chebsample(x->exp(a*im*x^2),2N)
        g = getchirp(i)
        h[string(N)] = g[:] = Fct.chebcoeffs(s)[1:2:N+1]
    end
    close(h)
end

function load(file)
    h=h5open(file,"r")
    for i in 1:L
        g = getchirp(i)
        g[:] = read(h, string(degrees[i]))
    end
    close(h)
end

__init__(args...) = println("Chirps.__init__($args)")

end #module

