#=
Lightweight implementation of forward-mode autodiff with complex number support.
=#
module Duals

import SpecialFunctions: erf, erfc, erfcinv, dawson
import Base: convert, promote_rule, +, -, *, /, inv, cos, sin, tan, cot, sec, csc, exp, cosh, sinh, tanh, coth, sech, csch, log, sqrt, one, zero, isnan, isinf, conj, real, imag

const erfinf=2/sqrt(pi)
const ierfinf=sqrt(pi)/2

#a+be with e^2=0
struct Dual{T<:Number} <: Number
	a::T
	b::T
end

Standard=Union{Real,Complex{<:Real}}

Dual(x::T) where T<:Number = Dual{T}(x, zero(T))

function Dual(x::T1,y::T2) where {T1<:Number,T2<:Number}
	T = promote_type(T1,T2)
	Dual{T}(promote(x,y)...)
end

function Dual(x::T1,y::T2,z::T3) where {T1<:Number,T2<:Number,T3<:Number}
	T = promote_type(T1,T2,T3)
	Dual{T}(promote(x,y,z)...)
end

convert(::Type{Dual},x::Standard)=Dual(x)
convert(::Type{Dual{T}},x::Standard) where T<:Number = Dual(convert(T,x)) 
convert(T::Type{<:Standard},x::Dual)=convert(T,x.a)

promote_rule(::Type{Dual{T1}}, ::Type{T2}) where {T1<:Number,T2<:Standard} = Dual{promote_type(T1,T2)}
promote_rule(::Type{Dual{T1}}, ::Type{Dual{T2}}) where {T1<:Number,T2<:Number} = Dual{promote_type(T1,T2)}

function (+)(x::Dual,y::Dual)
	Dual(x.a+y.a,x.b+y.b)
end

function (-)(x::Dual,y::Dual)
	Dual(x.a-y.a,x.b-y.b)
end

function (-)(y::Dual)
	Dual(-y.a,-y.b)
end

function (+)(x::Dual,y::Standard)
	Dual(x.a+y,x.b)
end

function (-)(x::Dual,y::Standard)
	Dual(x.a-y,x.b)
end

function (+)(x::Standard,y::Dual)
	Dual(x+y.a,y.b)
end

function (-)(x::Standard,y::Dual)
	Dual(x-y.a,-y.b)
end

function (*)(x::Standard,y::Dual)
	Dual(x*y.a,x*y.b)
end

function (*)(y::Dual,x::Standard)
	Dual(y.a*x,y.b*x)
end

function (/)(x::Dual,y::Standard)
	x*inv(y)
end

function (*)(x::Dual,y::Dual)
	Dual(x.a*y.a,x.a*y.b+x.b*y.a)
end

#x=z/y only if x*y=z
function (/)(z::Dual,y::Dual)
	r=inv(y.a)
	xa=r*z.a
	xb=r*(z.b-xa*y.b)
	Dual(xa,xb)
end

function inv(y::Dual)
	xa=inv(y.a)
	xb=-xa*xa*y.b
	Dual(xa,xb)	
end

function (/)(z::Standard,y::Dual)
	r=inv(y.a)
	xa=r*z
	xb=-r*xa*y.b
	Dual(xa,xb)
end

cos(x::Dual,si::Standard,co::Standard) = Dual(co, -x.b*si)
sin(x::Dual,si::Standard,co::Standard) = Dual(si, x.b*co)
tan(x::Dual,si::Standard,co::Standard) = sin(x,si,co) / cos(x,si,co)
cot(x::Dual,si::Standard,co::Standard) = cos(x,si,co) / sin(x,si,co)
sec(x::Dual,si::Standard,co::Standard) = inv(cos(x,si,co))
csc(x::Dual,si::Standard,co::Standard) = inv(sin(x,si,co))

cos(x::Dual)=cos(x,sincos(x.a)...)
sin(x::Dual)=sin(x,sincos(x.a)...)
tan(x::Dual)=tan(x,sincos(x.a)...)
cot(x::Dual)=cot(x,sincos(x.a)...)
sec(x::Dual)=sec(x,sincos(x.a)...)
csc(x::Dual)=csc(x,sincos(x.a)...)

function exp(x::Dual)
	ex=exp(x.a)
	Dual(ex, ex*x.b)
end

function log(x::Dual)
	Dual(log(x.a), x.b/x.a)
end

function sqrt(x::Dual)
	f=sqrt(x.a)
	Dual(f,x.b/2f)
end

function dawson(x::Dual)
	f=dawson(x.a)
	df=1-2*x.a*f
    Dual(f,x.b*df)
end

function erf(x::Dual)
	f=erf(x.a)
	df=erfinf*exp(-x.a*x.a)
    Dual(f,x.b*df)
end

function erfc(x::Dual)
	f=erfc(x.a)
	df=-erfinf*exp(-x.a*x.a)
	Dual(f,x.b*df)
end

function erfcinv(x::Dual)
	#erfcinv only defined on (0,2)
	f=erfcinv(real(x.a))
	df=-ierfinf*exp(f*f)
	Dual(f,x.b*df)
end

function cosh(x::Dual)
	p=exp(x)
	m=inv(p)
	(p+m)/2
end
function sinh(x::Dual)
	p=exp(x)
	m=inv(p)
	(p-m)/2
end
function tanh(x::Dual)
	p=exp(x)
	m=inv(p)
	(p-m)/(p+m)
end
function coth(x::Dual)
	p=exp(x)
	m=inv(p)
	(p+m)/(p-m)
end
function sech(x::Dual)
	p=exp(x)
	m=inv(p)
	2/(p+m)
end
function csch(x::Dual)
	p=exp(x)
	m=inv(p)
	2/(p-m)
end

function real(x::Dual)
	Dual(real(x.a),real(x.b))
end

function imag(x::Dual)
	Dual(imag(x.a),imag(x.b))
end

function conj(x::Dual)
	Dual(conj(x.a),conj(x.b))
end

one(::Dual{T}) where T<:Number = Dual(one(T))
zero(::Dual{T}) where T<:Number = Dual(zero(T))

isnan(x::Dual)=isnan(x.a) || isnan(x.b)
isinf(x::Dual)=isinf(x.a) || isinf(x.b)

#d(arg(z)) = imag(dz/z)
phase_velocity(z::Dual)=imag(z.b/z.a)


end #module
