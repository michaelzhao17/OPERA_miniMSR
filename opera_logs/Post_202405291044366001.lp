Opera-3d Post-Processor                 started on 29/May/2024 at 10:44:36
Opera 2021 SP3 [Build 3]
(c) Copyright Dassault Systemes Ltd 1984-2021. All Rights Reserved.
Node: UCNENG01. Processor: Intel64 Family 6 Model 42 Stepping 7, GenuineIntel. System: Windows 10 Version 2009 windows (10.0.19045)
Number of threads = 4. Available processors (CPU cores) = 4.
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
**** COMMAND INPUT *** ACTIVATE FILE='C:\Users\mzhao\OPERA_miniMSR\layer_mesh.op3'
Opening file for checking: C:\Users\mzhao\OPERA_miniMSR\layer_mesh.op3
**** COMMAND INPUT *** LOAD
Attaching file as resident: C:\Users\mzhao\OPERA_miniMSR\layer_mesh.op3
Opening database ...s\mzhao\OPERA_miniMSR\layer_mesh.op3, simulation number 1 on 29/May/2024 at 10:44:37
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
At (0.0,0.0,0.0), B = 1.40473154197863E-06
**** COMMAND INPUT ***
**** COMMAND INPUT *** LINE  BUFFERNAME='Line' X1=0 X2=X1 Y1=0 Y2=Y1 Z1=-1 Z2=1 NP=100
Coordinates (0.0,0.0,-1.0)
Failed to evaluate fields at 101 points.
Point outside finite element mesh.
Coordinates (0.0,0.0,-1.0)
**** COMMAND INPUT ***
**** COMMAND INPUT *** LINE  BUFFERNAME='Line' X1=0 X2=X1 Y1=0 Y2=Y1 Z1=-0.8 Z2=0.8 NP=100
**** COMMAND INPUT *** DATALINE OPTION=CREATE BUFFERNAME='Line' MINX=* MAXX=* NAME='NewLine' XCOMPONENT=Z YCOMPONENT=B INTERPOLATION=LINEAR GRAPH='Default'
Integral: 2.96933e-05
Maximum: 4.64676e-05 (X at maximum: -0.8)
Minimum: 1.40199e-06 (X at minimum: -0.128)
**** COMMAND INPUT ***
**** COMMAND INPUT *** LINE  BUFFERNAME='Line' X1=0 X2=X1 Y1=0.8 Z1=0 Z2=Z1 Y2=-0.8 NP=100
**** COMMAND INPUT *** DATALINE OPTION=CREATE BUFFERNAME='Line' MINX=* MAXX=* NAME='NewLine_1' XCOMPONENT=y YCOMPONENT=B INTERPOLATION=LINEAR GRAPH='Default'
Integral: -7.92576e-05
Maximum: 9.34247e-05 (X at maximum: 0.336)
Minimum: 1.40473e-06 (X at minimum: 0)
**** COMMAND INPUT ***
User requested application close.
Opera-3d Post-Processor                complete on 29/May/2024 at 10:47:28
                                                00:02:51.71 (00:00:08.10 cp)
