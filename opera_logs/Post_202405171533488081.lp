Opera-3d Post-Processor                 started on 17/May/2024 at 15:33:48
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
Opening database ...ers\mzhao\OPERA_miniMSR\mini_MSR.op3, simulation number 1 on 17/May/2024 at 15:33:48
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
**** COMMAND INPUT *** POINT XP=0 YP=0 ZP=0 COMPONENT=B
At (0.0,0.0,0.0), B = 6.73650086232986E-07
**** COMMAND INPUT ***
**** COMMAND INPUT *** LINE  BUFFERNAME='Line' X1=0 X2=X1 Y1=0 Y2=Y1 Z1=-0.5 Z2=0.5 NP=100
**** COMMAND INPUT *** AXESVIEW OPTION=CREATE SPECTRALPLOT=NO TYPE=Cartesian
You must specify a valid graph name.
**** COMMAND INPUT ***
**** COMMAND INPUT *** LINE  BUFFERNAME='Line' X1=0 X2=X1 Y1=0 Y2=Y1 Z1=-0.5 Z2=0.5 NP=100
**** COMMAND INPUT *** AXESVIEW OPTION=CREATE SPECTRALPLOT=NO TYPE=Cartesian GRAPH='graph1'
**** COMMAND INPUT *** DATALINE OPTION=CREATE BUFFERNAME='Line' MINX=* MAXX=* NAME='NewLine' XCOMPONENT=X YCOMPONENT=Y INTERPOLATION=LINEAR GRAPH='graph1'
Integral: 0
Maximum: 0 (X at maximum: 0)
Minimum: 0 (X at minimum: 0)
**** COMMAND INPUT ***
**** COMMAND INPUT *** THREED OPTION=GETVIEW
**** COMMAND INPUT ***
**** COMMAND INPUT *** THREED TYPE=SURFACE VECTOR=NO XORIGIN=0.184999999999998 YORIGIN=0.184999999999996 ZORIGIN=-4.61852778244065E-16 ROTX=27.3465002970375 ROTY=50.9228931317936 ROTZ=-15.7562479556202 XASPECT=1 YASPECT=1 ZASPECT=1 SIZE=0.370000000000004 FACETANGLE=10 PERSPECTIVE=YES LINECOLOUR=YES USEOUTERELEM=NO OPTION=SETVIEW
**** COMMAND INPUT ***
**** COMMAND INPUT *** SELECT ACTION=SELECT  OPTION=SURFACES SYMMETRY=Three_Quarters
**** COMMAND INPUT *** THREED OPTION=REFRESH
**** COMMAND INPUT ***
**** COMMAND INPUT *** WINDOW OUTLINE=NO
**** COMMAND INPUT ***
**** COMMAND INPUT *** LINE  BUFFERNAME='Line' X1=0 X2=X1 Y1=0 Y2=Y1 Z1=-0.5 Z2=0.5 NP=100
**** COMMAND INPUT *** DATALINE OPTION=CREATE BUFFERNAME='Line' MINX=* MAXX=* NAME='NewLine_1' XCOMPONENT=z YCOMPONENT=B INTERPOLATION=LINEAR GRAPH='Default'
Integral: 3.22086e-06
Maximum: 2.16574e-05 (X at maximum: -0.5)
Minimum: 6.83335e-08 (X at minimum: -0.32)
**** COMMAND INPUT ***
**** COMMAND INPUT *** LINE  BUFFERNAME='Line' X1=0 X2=X1 Y1=-0.5 Z1=0 Z2=Z1 Y2=0.5 NP=100
**** COMMAND INPUT *** DATALINE OPTION=CREATE BUFFERNAME='Line' MINX=* MAXX=* NAME='NewLine' XCOMPONENT=y YCOMPONENT=B INTERPOLATION=LINEAR GRAPH='Default'
Integral: 2.88954e-05
Maximum: 9.93602e-05 (X at maximum: -0.37)
Minimum: 6.7365e-07 (X at minimum: 0)
**** COMMAND INPUT ***
Opera-3d Post-Processor                complete on 17/May/2024 at 15:41:27
                                                00:07:39.31 (00:00:45.21 cp)
