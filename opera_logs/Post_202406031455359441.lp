Opera-3d Post-Processor                 started on 03/Jun/2024 at 14:55:35
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
**** COMMAND INPUT *** ACTIVATE FILE='C:\Users\mzhao\OPERA_miniMSR\miniMSR_layer.op3'
Opening file for checking: C:\Users\mzhao\OPERA_miniMSR\miniMSR_layer.op3
**** COMMAND INPUT *** LOAD
Attaching file as resident: C:\Users\mzhao\OPERA_miniMSR\miniMSR_layer.op3
Opening database ...zhao\OPERA_miniMSR\miniMSR_layer.op3, simulation number 1 on 03/Jun/2024 at 14:55:36
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
**** COMMAND INPUT *** POINT XP=0 YP=0 ZP=0 COMPONENT=X
At (0.0,0.0,0.0), X = 0.0
**** COMMAND INPUT ***
**** COMMAND INPUT *** LINE  BUFFERNAME='Line' X1=0 X2=X1 Y1=0 Y2=Y1 Z1=-0.4 Z2=0.4 NP=100
**** COMMAND INPUT *** DATALINE OPTION=CREATE BUFFERNAME='Line' MINX=* MAXX=* NAME='NewLine' XCOMPONENT=z YCOMPONENT=H INTERPOLATION=LINEAR GRAPH='Default'
Integral: 0.571291
Maximum: 6.64038 (X at maximum: 0.4)
Minimum: 0.203933 (X at minimum: -0.128)
**** COMMAND INPUT ***
**** COMMAND INPUT *** BUFFER NAME='Line' OPTION=SWITCH
**** COMMAND INPUT ***
**** COMMAND INPUT *** LINE  BUFFERNAME='Line' X1=0 X2=X1 Y1=0 Y2=Y1 Z1=-1.8 Z2=1.8 NP=100
**** COMMAND INPUT *** DATALINE OPTION=CREATE BUFFERNAME='Line' MINX=* MAXX=* NAME='NewLine_1' XCOMPONENT=z YCOMPONENT=H INTERPOLATION=LINEAR GRAPH='Default'
Integral: 110.901
Maximum: 47.4149 (X at maximum: -1.8)
Minimum: 0.203956 (X at minimum: -0.108)
**** COMMAND INPUT ***
**** COMMAND INPUT *** LINE  BUFFERNAME='Line' X1=0 X2=X1 Y1=1.8 Z1=0 Z2=Z1 Y2=-1.8 NP=100
**** COMMAND INPUT *** DATALINE OPTION=CREATE BUFFERNAME='Line' MINX=* MAXX=* NAME='NewLine_2' XCOMPONENT=Y YCOMPONENT=H INTERPOLATION=LINEAR GRAPH='Default'
Integral: -172.049
Maximum: 79.3424 (X at maximum: 0.396)
Minimum: 0.204598 (X at minimum: 0)
**** COMMAND INPUT ***
**** COMMAND INPUT *** LINE  BUFFERNAME='Line' X1=0 X2=X1 Y1=0.4 Z1=0 Z2=Z1 Y2=-0.4 NP=100
**** COMMAND INPUT *** DATALINE OPTION=CREATE BUFFERNAME='Line' MINX=* MAXX=* NAME='NewLine' XCOMPONENT=Y YCOMPONENT=H INTERPOLATION=LINEAR GRAPH='Default'
Integral: -6.46682
Maximum: 79.4228 (X at maximum: 0.376)
Minimum: 0.204598 (X at minimum: 0)
**** COMMAND INPUT ***
Opera-3d Post-Processor                complete on 03/Jun/2024 at 15:01:58
                                                00:06:23.33 (00:00:22.53 cp)
