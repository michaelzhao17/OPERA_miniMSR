Opera-3d Post-Processor                 started on 23/May/2024 at 12:53:18
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
**** COMMAND INPUT *** ACTIVATE FILE='C:\Users\mzhao\OPERA_miniMSR\mini_MSR.op3'
Opening file for checking: C:\Users\mzhao\OPERA_miniMSR\mini_MSR.op3
**** COMMAND INPUT *** LOAD
Attaching file as resident: C:\Users\mzhao\OPERA_miniMSR\mini_MSR.op3
Opening database ...ers\mzhao\OPERA_miniMSR\mini_MSR.op3, simulation number 1 on 23/May/2024 at 12:53:20
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
**** COMMAND INPUT *** WINDOW OUTLINE=NO
**** COMMAND INPUT ***
**** COMMAND INPUT *** POINT XP=0 YP=0 ZP=0 COMPONENT=B
At (0.0,0.0,0.0), B = 1.13677503775372E-06
**** COMMAND INPUT ***
**** COMMAND INPUT *** LINE  BUFFERNAME='Line' X1=0 X2=X1 Y1=0 Y2=Y1 Z1=0.5 Z2=-0.5 NP=100
**** COMMAND INPUT *** DATALINE OPTION=CREATE BUFFERNAME='Line' MINX=* MAXX=* NAME='NewLine' XCOMPONENT=Z YCOMPONENT=B INTERPOLATION=LINEAR GRAPH='Default'
Integral: -2.94602e-05
Maximum: 9.79468e-05 (X at maximum: -0.38)
Minimum: 1.12832e-06 (X at minimum: 0.03)
**** COMMAND INPUT ***
**** COMMAND INPUT *** LINE  BUFFERNAME='Line' X1=0 X2=X1 Y1=0 Y2=Y1 Z1=1 Z2=-1 NP=200
**** COMMAND INPUT *** DATALINE OPTION=CREATE BUFFERNAME='Line' MINX=* MAXX=* NAME='NewLine_1' XCOMPONENT=Z YCOMPONENT=B INTERPOLATION=LINEAR GRAPH='Default'
Integral: -0.000111364
Maximum: 9.79468e-05 (X at maximum: -0.38)
Minimum: 1.12832e-06 (X at minimum: 0.03)
**** COMMAND INPUT ***
**** COMMAND INPUT *** LINE  BUFFERNAME='Line' X1=0 X2=X1 Y1=1 Z1=1 Z2=Z1 Y2=-1 NP=200
**** COMMAND INPUT *** DATALINE OPTION=CREATE BUFFERNAME='Line' MINX=* MAXX=* NAME='NewLine_2' XCOMPONENT=Y YCOMPONENT=B INTERPOLATION=LINEAR GRAPH='Default'
Integral: -0.000136343
Maximum: 7.29888e-05 (X at maximum: 0.04)
Minimum: 6.3411e-05 (X at minimum: -1)
**** COMMAND INPUT ***
**** COMMAND INPUT *** LINE  BUFFERNAME='Line' X1=0 X2=X1 Y1=1 Z1=0 Z2=Z1 Y2=-1 NP=200
**** COMMAND INPUT *** DATALINE OPTION=CREATE BUFFERNAME='Line' MINX=* MAXX=* NAME='NewLine_3' XCOMPONENT=Y YCOMPONENT=B INTERPOLATION=LINEAR GRAPH='Default'
Integral: -4.7238e-05
Maximum: 5.4887e-05 (X at maximum: 1)
Minimum: 2.57628e-07 (X at minimum: -0.32)
**** COMMAND INPUT ***
User requested application close.
Opera-3d Post-Processor                complete on 24/May/2024 at 09:05:00
                                                20:11:41.80 (00:02:23.20 cp)
