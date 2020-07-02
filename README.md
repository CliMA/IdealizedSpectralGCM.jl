# IdealizedSpetralGCM.jl
JGCM is a lightweight educational spectral atmosphere model written with Julia. 
The code follows the classical spectral atmosphere model developed in Geophysical Fluid Dynamics Laboratory.

## Install `JGCM`
If you intend to develop the package (add new features, modify current functions, etc.), we suggest developing the package (in the current directory (GCM-from-Scratch))
```
julia> ]
pkg> dev .
```

You should be able to load the module 
```
julia> using JGCM
```

When necessary, yoGCMu can delete the package (in the current directory (GCM-from-Scratch))
```
julia> ]
pkg> rm JGCM
```

## Code structure
* The spectral method is in /src/Atmos_Spectral

* The parameterization, currently only the notable Held-Suarez test case, is in /src/Atmos_Param
  
* Experiments, including Barotropic flow, showllow water equations and Held-Suarez test case, are in /exp

## References
1. https://www.gfdl.noaa.gov/idealized-spectral-models-quickstart/
   
2. Ehrendorfer, Martin. Spectral numerical weather prediction models. Society for Industrial and Applied Mathematics, 2011. 
   
3. Durran, Dale R. Numerical methods for wave equations in geophysical fluid dynamics. Vol. 32. Springer Science & Business Media, 2013.
   
4. Lauritzen, Peter H., et al., eds. Numerical techniques for global atmospheric models. Vol. 80. Springer Science & Business Media, 2011.



