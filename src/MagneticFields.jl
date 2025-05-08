# Magnetic field functions
"""
  B_polynomial(α...)

Return an axially varying magnetic field that follows a polynomial function.
`α` contains the coefficients of the polynomial for increasing order.
"""
function B_polynomial(α...)
  _B(x) = sum(α .* [x^(i-1) for i in 1:length(α)]) 

  return _B
end
"""
  B_tanh(x₀, α, β, γ)

Return an axially varying magnetic field that follows a parametrized `tanh` function.
"""
function B_tanh(x₀, α, β, γ)
  _B(x) = (1 + α*tanh(γ*(β - abs(x - x₀))))/(1 + α*tanh(γ*β)) 

  return _B
end

"""
  B_tanh_MaPLE(x₀, α, β)

Return an axially varying magnetic field that follows a parametrized `tanh` function.

Arguments:
* `x₀`: location of field maximum value.
* `α`: distance to `x₀` where the field magnitude is half its maximum.
* `β`: changes the _flatness_ of the plateau around the maximum value.
"""
function B_tanh_MaPLE(x₀, α, β)
  _B(x) = 0.5*(1 - tanh((abs(x - x₀) - α)/β)) 

  return _B
end

"""
  B_arctan(x₀, α, β, γ)

Return an axially varying magnetic field that follows a parametrized `arctan` function.
"""
function B_arctan(x₀, α, β, γ)
  _B(x) = (1 + α*atan(γ*(β - abs(x - x₀))))/(1 + α*atan(γ*β))

  return _B
end
"""
 B_Moreau

Implicit field defined in R.Moreau et al. (2010) PMC Physics B 3(1):3 
"""
function B_Moreau(z₀)

  f(z) = β -> 3*(1-β[1])/(1+β[1]) - exp(4-2/β[1]-(z-z₀)*π)

  _B(z)=nlsolve(f(z),[0.01]).zero[1]

 return _B
end

# Field manipulation functions
"""
  curl_free_B(B)

Given `B`, a 1D magnetic field fit, it returns a function with a curl free correction
for a magnetic field consistent with real fields [1].

[1]: X. Albets-Chico et al. (2011), Fusion Eng. Des. 86(1), 5-14.
"""
function curl_free_B(B)
  dB(x) = ForwardDiff.derivative(B, x)
  d²B(x) = ForwardDiff.derivative(dB, x)
  d³B(x) = ForwardDiff.derivative(d²B, x)
  d⁴B(x) = ForwardDiff.derivative(d³B, x)

  function _curl_free_B(x)
    B₁ = 0.0
    B₂ = B(x[3]) - d²B(x[3])*x[2]^2/2 + d⁴B(x[3])*x[2]^4/24
    B₃ = dB(x[3])*x[2] - d³B(x[3])*x[2]^3/6

    return VectorValue(B₁, B₂, B₃)
  end

  return _curl_free_B
end
