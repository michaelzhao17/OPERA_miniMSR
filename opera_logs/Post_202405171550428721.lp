Opera-3d Post-Processor                 started on 17/May/2024 at 15:50:42
Opera 2020 SP3 [Build 2]
(c) Copyright Dassault Systemes Ltd 1984-2019. All Rights Reserved.
Node: TUCAN-MAGPC. Processor: Intel64. System: Windows 10
Number of threads = 8. Available processors (CPU cores) = 8.
Command language initialization.
Unit of Length           : cm                | Unit of Magn Flux Density: gauss
Unit of Magnetic Field   : oersted           | Unit of Magn Scalar Pot  : oersted cm
Unit of Magn Vector Pot  : gauss cm          | Unit of Conductivity     : S/cm
Unit of Current Density  : A/cm^2            | Unit of Power            : W
Unit of Force            : N                 | Unit of Energy           : J
Unit of Electric Field   : V/cm              | Unit of Elec Flux Density: C/cm^2
Unit of Mass             : g                 | Unit of Pressure         : Pa
Unit of Charge Density   : C/cm^3            | Unit of Electric Pot     : volt
Information: opera.comi was not found in the local or home directory
**** COMMAND INPUT *** ACTIVATE FILE='C:\Users\mzhao\OPERA_miniMSR\mini_MSR.op3'
Opening file for checking: C:\Users\mzhao\OPERA_miniMSR\mini_MSR.op3
**** COMMAND INPUT *** LOAD
Attaching file as resident: C:\Users\mzhao\OPERA_miniMSR\mini_MSR.op3
Opening database ...ers\mzhao\OPERA_miniMSR\mini_MSR.op3, simulation number 1 on 17/May/2024 at 15:50:43
Unit of Length           : m                 | Unit of Magn Flux Density: T
Unit of Magnetic Field   : A/m               | Unit of Magn Scalar Pot  : A
Unit of Magn Vector Pot  : Wb/m              | Unit of Conductivity     : S/m
Unit of Current Density  : A/m^2             | Unit of Power            : W
Unit of Force            : N                 | Unit of Energy           : J
Unit of Electric Field   : V/m               | Unit of Elec Flux Density: C/m^2
Unit of Mass             : kg                | Unit of Pressure         : Pa
Unit of Charge Density   : C/m^3             | Unit of Electric Pot     : volt
**** COMMAND INPUT *** SELECT ACTION=DEFAULT
**** COMMAND INPUT *** SELECT ACTION=SELECT OPTION=SURFACES
**** COMMAND INPUT *** THREED OPTION=REFRESH
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT *** WINDOW SOLID=NO
**** COMMAND INPUT ***
**** COMMAND INPUT *** WINDOW SOLID=YES
**** COMMAND INPUT ***
**** COMMAND INPUT *** WINDOW OUTLINE=NO
**** COMMAND INPUT ***
**** COMMAND INPUT *** WINDOW SOLID=NO
**** COMMAND INPUT ***
**** COMMAND INPUT *** WINDOW SOLID=YES
**** COMMAND INPUT ***
**** COMMAND INPUT *** LINE  BUFFERNAME='Line' X1=2 Y1=0 Y2=Y1 Z1=0 Z2=Z1 X2=-2 NP=200
**** COMMAND INPUT *** DATALINE OPTION=CREATE BUFFERNAME='Line' MINX=* MAXX=* NAME='NewLine' XCOMPONENT=X YCOMPONENT=B INTERPOLATION=LINEAR GRAPH='Default'
Integral: -0.0001639
Maximum: 6.02843e-05 (X at maximum: 2)
Minimum: 7.88676e-08 (X at minimum: 0.32)
**** COMMAND INPUT ***
**** COMMAND INPUT *** LINE  BUFFERNAME='Line' X1=2 X2=X1 Y1=2 Z1=0 Z2=Z1 Y2=-2 NP=200
**** COMMAND INPUT *** DATALINE OPTION=CREATE BUFFERNAME='Line' MINX=* MAXX=* NAME='NewLine_1' XCOMPONENT=y YCOMPONENT=B INTERPOLATION=LINEAR GRAPH='Default'
Integral: -0.000246237
Maximum: 6.27776e-05 (X at maximum: 2)
Minimum: 6.0282e-05 (X at minimum: -0.02)
**** COMMAND INPUT ***
**** COMMAND INPUT *** LINE  BUFFERNAME='Line' X1=0 X2=X1 Y1=2 Z1=0 Z2=Z1 Y2=-2 NP=200
**** COMMAND INPUT *** DATALINE OPTION=CREATE BUFFERNAME='Line' MINX=* MAXX=* NAME='NewLine_2' XCOMPONENT=y YCOMPONENT=B INTERPOLATION=LINEAR GRAPH='Default'
Integral: -0.000243638
Maximum: 9.95098e-05 (X at maximum: 0.38)
Minimum: 6.67102e-07 (X at minimum: -0.02)
**** COMMAND INPUT ***
**** COMMAND INPUT *** THREED OPTION=GETVIEW
**** COMMAND INPUT ***
**** COMMAND INPUT *** THREED TYPE=COMPONENT COMPONENT=Bz MIN=* MAX=* VECTOR=NO XORIGIN=0 YORIGIN=-3.5527136788005E-17 ZORIGIN=1.4210854715202E-16 ROTX=44.5058766022238 ROTY=62.6949946354153 ROTZ=-31.9947026113157 XASPECT=1 YASPECT=1 ZASPECT=1 SIZE=0.434721444583522 FACETANGLE=10 PERSPECTIVE=YES LINECOLOUR=YES USEOUTERELEM=NO OPTION=SETVIEW
Min (-0.0114689907503237) at (-0.369999999999995,-4.32989280598354E-03,0.337499594375188), Max (0.0118898989300355) at (-0.369999999999995,-2.5003743663099E-03,-0.335670155208731)
**** COMMAND INPUT ***
**** COMMAND INPUT *** THREED OPTION=GETVIEW
**** COMMAND INPUT ***
**** COMMAND INPUT *** THREED TYPE=SURFACE VECTOR=YES VX=Bx VY=By VZ=Bz VMIN=* VMAX=* VECSBYSIZE=YES VSCALEOPTION=AUTOMATIC VAUTOSCALE=1 XORIGIN=0 YORIGIN=-3.5527136788005E-17 ZORIGIN=1.4210854715202E-16 ROTX=10.3247271640374 ROTY=64.8665353386001 ROTZ=-0.487133430311238 XASPECT=1 YASPECT=1 ZASPECT=1 SIZE=0.617304451308601 FACETANGLE=10 PERSPECTIVE=YES LINECOLOUR=YES USEOUTERELEM=NO OPTION=SETVIEW
**** COMMAND INPUT ***
**** COMMAND INPUT *** THREED OPTION=GETVIEW
**** COMMAND INPUT ***
**** COMMAND INPUT *** THREED TYPE=SURFACE VECTOR=YES VMIN=* VMAX=* VECSBYSIZE=YES VSCALEOPTION=AUTOMATIC VAUTOSCALE=1 XORIGIN=0 YORIGIN=-3.5527136788005E-17 ZORIGIN=1.4210854715202E-16 ROTX=7.11677895844934 ROTY=44.3201924036362 ROTZ=-2.03134205251711 XASPECT=1 YASPECT=1 ZASPECT=1 SIZE=0.436844956451285 FACETANGLE=10 PERSPECTIVE=YES LINECOLOUR=YES USEOUTERELEM=NO OPTION=SETVIEW
**** COMMAND INPUT ***
**** COMMAND INPUT *** THREED OPTION=GETVIEW
**** COMMAND INPUT ***
**** COMMAND INPUT *** THREED TYPE=SURFACE VECTOR=NO XORIGIN=0 YORIGIN=-3.5527136788005E-17 ZORIGIN=1.4210854715202E-16 ROTX=7.11677895844934 ROTY=44.3201924036362 ROTZ=-2.03134205251711 XASPECT=1 YASPECT=1 ZASPECT=1 SIZE=0.436844956451285 FACETANGLE=10 PERSPECTIVE=YES LINECOLOUR=YES USEOUTERELEM=NO OPTION=SETVIEW
**** COMMAND INPUT ***
**** COMMAND INPUT *** LINE  BUFFERNAME='Line' X1=0 X2=X1 Y1=0 Y2=Y1 Z1=-2 Z2=2 NP=200
**** COMMAND INPUT *** DATALINE OPTION=CREATE BUFFERNAME='Line' MINX=* MAXX=* NAME='NewLine_3' XCOMPONENT=z YCOMPONENT=B INTERPOLATION=LINEAR GRAPH='Default'
Integral: 0.000163796
Maximum: 6.02454e-05 (X at maximum: 2)
Minimum: 8.60148e-08 (X at minimum: -0.32)
**** COMMAND INPUT ***
Opera-3d Post-Processor                complete on 17/May/2024 at 15:56:43
                                                00:06:01.34 (00:01:25.57 cp)
