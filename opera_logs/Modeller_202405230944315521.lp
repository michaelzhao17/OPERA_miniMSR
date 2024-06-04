Opera-3d Modeller                       started on 23/May/2024 at 09:44:31
Opera 2021 SP3 [Build 3]
(c) Copyright Dassault Systemes Ltd 1984-2021. All Rights Reserved.
Node: UCNENG01. Processor: Intel64 Family 6 Model 42 Stepping 7, GenuineIntel. System: Windows 10 Version 2009 windows (10.0.19045)
Command language initialization.
Information: modeller.comi was not found in the local or home directory
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=UNPICK REDISPLAY=YES
**** COMMAND INPUT ***
*** CONTROL COMMAND ** $ COMI 'C:\Users\mzhao\OPERA_miniMSR\test.comi'
Opening file for input: C:\Users\mzhao\OPERA_miniMSR\test.comi
*** COMMENT ********** // this script creates model of mini-MSR in OPERA
*** COMMENT ********** // author: M. Zhao
*** COMMENT ********** // May 2024
*** CONTROL COMMAND ** $string yesorno YES
 yesorno='YES'
**** FILE INPUT ****** CLEAR REVERT=NO
Command language initialization.
**** FILE INPUT ****** THREED XORIGIN=0 YORIGIN=0 ZORIGIN=0 ROTX=0 ROTY=0 ROTZ=0 XASPECT=1 YASPECT=1 ZASPECT=1 SIZE=0 FACETANGLE=5 PERSPECTIVE=YES LINECOLOUR=YES OPTION=SETVIEW
**** FILE INPUT ******
*** COMMENT ********** ////////////////////////////////////////////////////////////
*** COMMENT ********** / Constants
*** COMMENT ********** ////////////////////////////////////////////////////////////
*** COMMENT ********** //Data Storage Levels
*** COMMENT ********** / mumetal
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#mu_DSL, Value=500, DESCRIPTION='data storage level for mumetal'
Variable Value            Expression
#mu_DSL  500.0            500
Description: data storage level for mumetal
*** COMMENT ********** / air (total potential)
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#air_tot_DSL, Value=200, DESCRIPTION='data storage level for total potential air'
Variable     Value            Expression
#air_tot_DSL 200.0            200
Description: data storage level for total potential air
*** COMMENT ********** / air (reduced potential)
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#air_red_DSL, Value=100, DESCRIPTION='data storage level for reduced potential air'
Variable     Value            Expression
#air_red_DSL 100.0            100
Description: data storage level for reduced potential air
**** FILE INPUT ******
*** COMMENT ********** /// meshing parameters
*** COMMENT ********** // max element size (MES)
*** COMMENT ********** / flat mumetal
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#mu_flat_MES, Value=0.05, DESCRIPTION='maximum element size for flat mumetal'
Variable     Value            Expression
#mu_flat_MES 0.05             0.05
Description: maximum element size for flat mumetal
*** COMMENT ********** / bent mumetal
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#mu_bent_MES, Value=0.03, DESCRIPTION='maximum element size for bent mumetal'
Variable     Value            Expression
#mu_bent_MES 0.03             0.03
Description: maximum element size for bent mumetal
*** COMMENT ********** / air (total potential)
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#air_tot_MES, Value=0.02, DESCRIPTION='maximum element size for total potential air'
Variable     Value            Expression
#air_tot_MES 0.02             0.02
Description: maximum element size for total potential air
*** COMMENT ********** / air (reduced potential)
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#air_red_MES, Value=0.3, DESCRIPTION='maximum element size for reduced potential air'
Variable     Value            Expression
#air_red_MES 0.3              0.3
Description: maximum element size for reduced potential air
**** FILE INPUT ******
*** COMMENT ********** // maximum angle between elements (MAbE)
*** COMMENT ********** / flat mumetal
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#mu_flat_MAbE, Value=30, DESCRIPTION='maximum angle between elements for flat mumetal'
Variable      Value            Expression
#mu_flat_MAbE 30.0             30
Description: maximum angle between elements for flat mumetal
*** COMMENT ********** / bent mumetal
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#mu_bent_MAbE, Value=30, DESCRIPTION='maximum angle between elements for bent mumetal'
Variable      Value            Expression
#mu_bent_MAbE 30.0             30
Description: maximum angle between elements for bent mumetal
*** COMMENT ********** / air
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#air_MAbE, Value=30, DESCRIPTION='maximum angle between elements for air'
Variable      Value            Expression
#air_MAbE     30.0             30
Description: maximum angle between elements for air
**** FILE INPUT ******
**** FILE INPUT ******
*** COMMENT ********** ////////////////////////////////////////////////////////////
*** COMMENT ********** / Modelling
*** COMMENT ********** ////////////////////////////////////////////////////////////
**** FILE INPUT ******
*** COMMENT ********** //// Inner Layer
**** FILE INPUT ******
*** COMMENT ********** / load structural step file
**** FILE INPUT ****** LOAD OPTION=INSERT FILE='stpfiles//inner_layer.STEP'
Imported material name from label: layer1_NAUO1#inner_flat_front_Boss_Extrude2
Imported material name from label: layer1_NAUO2#inner_bracket_Cut_Extrude3
Imported material name from label: layer1_NAUO3#inner_bracket_Cut_Extrude3
Imported material name from label: layer1_NAUO4#inner_bracket_Cut_Extrude3
Imported material name from label: layer1_NAUO5#inner_bracket_Cut_Extrude3
Imported material name from label: layer1_NAUO6#inner_flat_Boss_Extrude2
Imported material name from label: layer1_NAUO7#inner_flat_Boss_Extrude2
Imported material name from label: layer1_NAUO8#inner_bracket_Cut_Extrude3
Imported material name from label: layer1_NAUO9#inner_bracket_Cut_Extrude3
Imported material name from label: layer1_NAUO10#inner_flat_Boss_Extrude2
Imported material name from label: layer1_NAUO11#inner_bracket_Cut_Extrude3
Imported material name from label: layer1_NAUO12#inner_bracket_Cut_Extrude3
Imported material name from label: layer1_NAUO13#inner_bracket_Cut_Extrude3
Imported material name from label: layer1_NAUO14#inner_bracket_Cut_Extrude3
Imported material name from label: layer1_NAUO15#inner_bracket_Cut_Extrude3
Imported material name from label: layer1_NAUO16#inner_bracket_Cut_Extrude3
Imported material name from label: layer1_NAUO17#inner_flat_Boss_Extrude2
Imported material name from label: layer1_NAUO18#inner_flat_Boss_Extrude2
**** FILE INPUT ******
*** COMMENT ********** /// Flat panels
*** COMMENT ********** // front
*** COMMENT ********** / rename body
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-001'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_front' UNIQUENAME='inner_front'
*** COMMENT ********** / set material to air
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_front'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=#air_tot_DSL SIZE=#air_tot_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / face settings
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** / set face to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_front' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_flat_MES NORMALTOL=#mu_flat_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
*** COMMENT ********** // back
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-018'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_back' UNIQUENAME='inner_back'
*** COMMENT ********** / set material to air
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_back'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=#air_tot_DSL SIZE=#air_tot_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / face settings
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** / set face to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_back' IDENTIFIER=A.00010
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_flat_MES NORMALTOL=#mu_flat_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
*** COMMENT ********** // left
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-010'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_left' UNIQUENAME='inner_left'
*** COMMENT ********** / set material to air
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_left'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=#air_tot_DSL SIZE=#air_tot_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / face settings
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** / set face to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_left' IDENTIFIER=A.00010
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_flat_MES NORMALTOL=#mu_flat_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
*** COMMENT ********** // right
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-006'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_right' UNIQUENAME='inner_right'
*** COMMENT ********** / set material to air
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_right'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=#air_tot_DSL SIZE=#air_tot_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / face settings
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** / set face to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_right' IDENTIFIER=A.00010
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_flat_MES NORMALTOL=#mu_flat_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
*** COMMENT ********** // top
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-017'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_top' UNIQUENAME='inner_top'
*** COMMENT ********** / set material to air
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_top'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=#air_tot_DSL SIZE=#air_tot_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / face settings
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** / set face to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_top' IDENTIFIER=A.00010
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_floor' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_flat_MES NORMALTOL=#mu_flat_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
*** COMMENT ********** // bottom
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-007'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_bottom' UNIQUENAME='inner_bottom'
*** COMMENT ********** / set material to air
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_bottom'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=#air_tot_DSL SIZE=#air_tot_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / face settings
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** / set face to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_bottom' IDENTIFIER=A.00010
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_floor' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_flat_MES NORMALTOL=#mu_flat_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ******
*** COMMENT ********** /// brackets
*** COMMENT ********** // front-top
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-005'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_brkt_front_top' UNIQUENAME='inner_brkt_front_top'
*** COMMENT ********** / set material to air
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_brkt_front_top'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=#air_tot_DSL SIZE=#air_tot_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / face settings
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** / set face to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_front_top' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_front_top' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_front_top' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
*** COMMENT ********** // front-bottom
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-003'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_brkt_front_bottom' UNIQUENAME='inner_brkt_front_bottom'
*** COMMENT ********** / set material to air
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_brkt_front_bottom'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=#air_tot_DSL SIZE=#air_tot_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / face settings
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** / set face to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_front_bottom' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_front_bottom' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_front_bottom' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
*** COMMENT ********** // front-left
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-004'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_brkt_front_left' UNIQUENAME='inner_brkt_front_left'
*** COMMENT ********** / set material to air
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_brkt_front_left'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=#air_tot_DSL SIZE=#air_tot_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / face settings
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** / set face to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_front_left' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_front_left' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_front_left' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
*** COMMENT ********** // front-right
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-002'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_brkt_front_right' UNIQUENAME='inner_brkt_front_right'
*** COMMENT ********** / set material to air
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_brkt_front_right'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=#air_tot_DSL SIZE=#air_tot_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / face settings
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** / set face to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_front_right' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_front_right' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_front_right' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
*** COMMENT ********** // bottom-left
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-012'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_brkt_bottom-left' UNIQUENAME='inner_brkt_bottom-left'
*** COMMENT ********** / set material to air
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_brkt_bottom-left'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=#air_tot_DSL SIZE=#air_tot_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / face settings
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** / set face to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_bottom-left' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_floor' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_bottom-left' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_floor' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_bottom-left' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_floor' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
*** COMMENT ********** // bottom-right
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-008'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_brkt_bottom_right' UNIQUENAME='inner_brkt_bottom_right'
*** COMMENT ********** / set material to air
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_brkt_bottom_right'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=#air_tot_DSL SIZE=#air_tot_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / face settings
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** / set face to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_bottom_right' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_floor' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_bottom_right' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_floor' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_bottom_right' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_floor' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
*** COMMENT ********** // top-left
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-011'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_brkt_top_left' UNIQUENAME='inner_brkt_top_left'
*** COMMENT ********** / set material to air
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_brkt_top_left'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=#air_tot_DSL SIZE=#air_tot_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / face settings
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** / set face to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_top_left' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_floor' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_top_left' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_floor' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_top_left' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_floor' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
*** COMMENT ********** // top-right
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-009'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_brkt_top_right' UNIQUENAME='inner_brkt_top_right'
*** COMMENT ********** / set material to air
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_brkt_top_right'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=#air_tot_DSL SIZE=#air_tot_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / face settings
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** / set face to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_top_right' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_floor' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_top_right' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_floor' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_top_right' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_floor' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
*** COMMENT ********** // back-bottom
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-013'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_brkt_back_bottom' UNIQUENAME='inner_brkt_back_bottom'
*** COMMENT ********** / set material to air
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_brkt_back_bottom'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=#air_tot_DSL SIZE=#air_tot_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / face settings
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** / set face to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_back_bottom' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_back_bottom' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_back_bottom' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
*** COMMENT ********** // back-top
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-015'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_brkt_back_top' UNIQUENAME='inner_brkt_back_top'
*** COMMENT ********** / set material to air
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_brkt_back_top'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=#air_tot_DSL SIZE=#air_tot_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / face settings
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** / set face to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_back_top' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_back_top' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_back_top' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
*** COMMENT ********** // back-left
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-016'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_brkt_back_left' UNIQUENAME='inner_brkt_back_left'
*** COMMENT ********** / set material to air
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_brkt_back_left'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=#air_tot_DSL SIZE=#air_tot_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / face settings
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** / set face to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_back_left' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_back_left' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_back_left' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
*** COMMENT ********** // back-right
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-014'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_brkt_back_right' UNIQUENAME='inner_brkt_back_right'
*** COMMENT ********** / set material to air
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_brkt_back_right'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=#air_tot_DSL SIZE=#air_tot_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / face settings
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** / set face to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_back_right' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_back_right' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_back_right' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ******
**** FILE INPUT ******
*** COMMENT ********** // BH data
*** COMMENT ********** / create labels
**** FILE INPUT ****** BHDATA OPTION=NEW  LABEL='innermumetal_door_BH'
**** FILE INPUT ****** BHDATA OPTION=NEW  LABEL='innermumetal_floor_BH'
**** FILE INPUT ******
*** COMMENT ********** / assign bh file to label
**** FILE INPUT ****** BHDATA OPTION=LOAD LABEL=innermumetal_door_BH FILE='bhdata//miniMSR_inner_door_anh.bh'
BHDATA label innermumetal_door_BH loaded with data from the file bhdata//miniMSR_inner_door_anh.bh
**** FILE INPUT ****** BHDATA OPTION=LOAD LABEL=innermumetal_floor_BH FILE='bhdata//miniMSR_inner_floor_anh.bh'
BHDATA label innermumetal_floor_BH loaded with data from the file bhdata//miniMSR_inner_floor_anh.bh
**** FILE INPUT ******
*** COMMENT ********** / setup boundary condition
**** FILE INPUT ****** BOUNDARY UNPICK
**** FILE INPUT ****** BOUNDARY GUIINIT
**** FILE INPUT ****** BOUNDARY PICK 'inner_mumetal_door'
**** FILE INPUT ****** BOUNDARY OPTION=MODIFY CONDITION=THINPLATE PLATEMATERIAL=inner_mumetal_door PLATETHICKNESS=0.0015
**** FILE INPUT ****** BOUNDARY UNPICK
**** FILE INPUT ******
**** FILE INPUT ****** BOUNDARY UNPICK
**** FILE INPUT ****** BOUNDARY GUIINIT
**** FILE INPUT ****** BOUNDARY PICK 'inner_mumetal_floor'
**** FILE INPUT ****** BOUNDARY OPTION=MODIFY CONDITION=THINPLATE PLATEMATERIAL=inner_mumetal_floor PLATETHICKNESS=0.0015
**** FILE INPUT ****** BOUNDARY UNPICK
**** FILE INPUT ******
*** COMMENT ********** / match material label to bh label
**** FILE INPUT ****** MATERIALS UNPICK
**** FILE INPUT ****** MATERIALS GUIINIT
**** FILE INPUT ****** MATERIALS PICK 'inner_mumetal_floor'
**** FILE INPUT ****** MATERIALS OPTION=MODIFY MULINEARITY=NONLINEAR MUANISOTROPY=ISOTROPIC BH='innermumetal_floor_BH'
**** FILE INPUT ****** MATERIALS UNPICK
**** FILE INPUT ******
**** FILE INPUT ******
*** COMMENT ********** //// Outer Layer
**** FILE INPUT ******
*** COMMENT ********** / load structural step file
**** FILE INPUT ****** LOAD OPTION=INSERT FILE='stpfiles//outer_layer.STEP'
Imported material name from label: outer_layer_NAUO1#outer_flat_front_Boss_Extrude2
Imported material name from label: outer_layer_NAUO2#outer_bracket_Cut_Extrude4
Imported material name from label: outer_layer_NAUO3#outer_bracket_Cut_Extrude4
Imported material name from label: outer_layer_NAUO4#outer_flat_Boss_Extrude2
Imported material name from label: outer_layer_NAUO5#outer_flat_Boss_Extrude2
Imported material name from label: outer_layer_NAUO6#outer_bracket_Cut_Extrude4
Imported material name from label: outer_layer_NAUO7#outer_bracket_Cut_Extrude4
Imported material name from label: outer_layer_NAUO8#outer_bracket_Cut_Extrude4
Imported material name from label: outer_layer_NAUO9#outer_bracket_Cut_Extrude4
Imported material name from label: outer_layer_NAUO10#outer_flat_Boss_Extrude2
Imported material name from label: outer_layer_NAUO11#outer_bracket_Cut_Extrude4
Imported material name from label: outer_layer_NAUO12#outer_bracket_Cut_Extrude4
Imported material name from label: outer_layer_NAUO13#outer_bracket_Cut_Extrude4
Imported material name from label: outer_layer_NAUO14#outer_flat_Boss_Extrude2
Imported material name from label: outer_layer_NAUO15#outer_bracket_Cut_Extrude4
Imported material name from label: outer_layer_NAUO16#outer_bracket_Cut_Extrude4
Imported material name from label: outer_layer_NAUO17#outer_bracket_Cut_Extrude4
Imported material name from label: outer_layer_NAUO18#outer_flat_Boss_Extrude2
Imported material name from label: outer_layer_NAUO19#outer_bracket_Cut_Extrude4
Imported material name from label: outer_layer_NAUO20#outer_bracket_Cut_Extrude4
Imported material name from label: outer_layer_NAUO21#outer_bracket_Cut_Extrude4
Imported material name from label: outer_layer_NAUO22#outer_bracket_Cut_Extrude4
Imported material name from label: outer_layer_NAUO23#outer_bracket_Cut_Extrude4
Imported material name from label: outer_layer_NAUO24#outer_bracket_Cut_Extrude4
Imported material name from label: outer_layer_NAUO25#outer_flat_Boss_Extrude2
Imported material name from label: outer_layer_NAUO26#outer_flat_Boss_Extrude2
**** FILE INPUT ******
*** COMMENT ********** /// Flat panels
*** COMMENT ********** // front
*** COMMENT ********** / rename body
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-001'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='outer_front' UNIQUENAME='outer_front'
*** COMMENT ********** / set material to air
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_front'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=#air_tot_DSL SIZE=#air_tot_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / face settings
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** / set face to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_front' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_flat_MES NORMALTOL=#mu_flat_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
*** COMMENT ********** // back
*** COMMENT ********** / rename body
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-026'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='outer_back' UNIQUENAME='outer_back'
*** COMMENT ********** / set material to air
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_back'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=#air_tot_DSL SIZE=#air_tot_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / face settings
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** / set face to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_back' IDENTIFIER=A.00010
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_flat_MES NORMALTOL=#mu_flat_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
*** COMMENT ********** // left
*** COMMENT ********** / rename body
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-014'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='outer_left' UNIQUENAME='outer_left'
*** COMMENT ********** / set material to air
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_left'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=#air_tot_DSL SIZE=#air_tot_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / face settings
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** / set face to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_left' IDENTIFIER=A.00010
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_flat_MES NORMALTOL=#mu_flat_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
*** COMMENT ********** // right
*** COMMENT ********** / rename body
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-018'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='outer_right' UNIQUENAME='outer_right'
*** COMMENT ********** / set material to air
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_right'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=#air_tot_DSL SIZE=#air_tot_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / face settings
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** / set face to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_right' IDENTIFIER=A.00010
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_flat_MES NORMALTOL=#mu_flat_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
*** COMMENT ********** // top
*** COMMENT ********** / rename body
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-025'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='outer_top' UNIQUENAME='outer_top'
*** COMMENT ********** / set material to air
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_top'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=#air_tot_DSL SIZE=#air_tot_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / face settings
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** / set face to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_top' IDENTIFIER=A.00010
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_floor' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_flat_MES NORMALTOL=#mu_flat_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
*** COMMENT ********** // bottom
*** COMMENT ********** / rename body
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-005'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='outer_bottom' UNIQUENAME='outer_bottom'
*** COMMENT ********** / set material to air
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_bottom'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=#air_tot_DSL SIZE=#air_tot_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / face settings
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** / set face to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_bottom' IDENTIFIER=A.00010
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_floor' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_flat_MES NORMALTOL=#mu_flat_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
*** COMMENT ********** /// brackets
*** COMMENT ********** // front-top
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-021'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='outer_brkt_front_top' UNIQUENAME='outer_brkt_front_top'
*** COMMENT ********** / set material to air
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_brkt_front_top'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=#air_tot_DSL SIZE=#air_tot_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / face settings
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** / set face to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_front_top' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_front_top' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_front_top' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
*** COMMENT ********** // front-bottom
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-003'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='outer_brkt_front_bottom' UNIQUENAME='outer_brkt_front_bottom'
*** COMMENT ********** / set material to air
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_brkt_front_bottom'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=#air_tot_DSL SIZE=#air_tot_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / face settings
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** / set face to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_front_bottom' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_front_bottom' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_front_bottom' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
*** COMMENT ********** // front-left
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-013'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='outer_brkt_front_left' UNIQUENAME='outer_brkt_front_left'
*** COMMENT ********** / set material to air
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_brkt_front_left'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=#air_tot_DSL SIZE=#air_tot_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / face settings
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** / set face to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_front_left' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_front_left' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_front_left' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
*** COMMENT ********** // front-right
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-017'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='outer_brkt_front_right' UNIQUENAME='outer_brkt_front_right'
*** COMMENT ********** / set material to air
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_brkt_front_right'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=#air_tot_DSL SIZE=#air_tot_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / face settings
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** / set face to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_front_right' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_front_right' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_front_right' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
*** COMMENT ********** // bottom-left
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-015'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='outer_brkt_bottom-left' UNIQUENAME='outer_brkt_bottom-left'
*** COMMENT ********** / set material to air
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_brkt_bottom-left'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=#air_tot_DSL SIZE=#air_tot_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / face settings
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** / set face to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_bottom-left' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_floor' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_bottom-left' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_floor' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_bottom-left' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_floor' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
*** COMMENT ********** // bottom-right
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-019'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='outer_brkt_bottom_right' UNIQUENAME='outer_brkt_bottom_right'
*** COMMENT ********** / set material to air
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_brkt_bottom_right'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=#air_tot_DSL SIZE=#air_tot_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / face settings
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** / set face to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_bottom_right' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_floor' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_bottom_right' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_floor' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_bottom_right' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_floor' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
*** COMMENT ********** // top-left
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-022'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='outer_brkt_top_left' UNIQUENAME='outer_brkt_top_left'
*** COMMENT ********** / set material to air
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_brkt_top_left'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=#air_tot_DSL SIZE=#air_tot_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / face settings
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** / set face to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_top_left' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_floor' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_top_left' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_floor' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_top_left' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_floor' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
*** COMMENT ********** // top-right
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-024'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='outer_brkt_top_right' UNIQUENAME='outer_brkt_top_right'
*** COMMENT ********** / set material to air
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_brkt_top_right'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=#air_tot_DSL SIZE=#air_tot_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / face settings
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** / set face to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_top_right' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_floor' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_top_right' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_floor' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_top_right' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_floor' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
*** COMMENT ********** // back-bottom
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-020'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='outer_brkt_back_bottom' UNIQUENAME='outer_brkt_back_bottom'
*** COMMENT ********** / set material to air
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_brkt_back_bottom'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=#air_tot_DSL SIZE=#air_tot_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / face settings
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** / set face to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_back_bottom' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_back_bottom' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_back_bottom' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
*** COMMENT ********** // back-top
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-023'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='outer_brkt_back_top' UNIQUENAME='outer_brkt_back_top'
*** COMMENT ********** / set material to air
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_brkt_back_top'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=#air_tot_DSL SIZE=#air_tot_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / face settings
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** / set face to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_back_top' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_back_top' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_back_top' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
*** COMMENT ********** // back-left
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-012'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='outer_brkt_back_left' UNIQUENAME='outer_brkt_back_left'
*** COMMENT ********** / set material to air
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_brkt_back_left'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=#air_tot_DSL SIZE=#air_tot_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / face settings
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** / set face to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_back_left' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_back_left' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_back_left' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
*** COMMENT ********** // back-right
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-016'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='outer_brkt_back_right' UNIQUENAME='outer_brkt_back_right'
*** COMMENT ********** / set material to air
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_brkt_back_right'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=#air_tot_DSL SIZE=#air_tot_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / face settings
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** / set face to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_back_right' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_back_right' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_back_right' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ******
*** COMMENT ********** // BH data
*** COMMENT ********** / create label
**** FILE INPUT ****** BHDATA OPTION=NEW  LABEL='outermumetal_door_BH'
**** FILE INPUT ****** BHDATA OPTION=NEW  LABEL='outermumetal_floor_BH'
**** FILE INPUT ******
*** COMMENT ********** / assign bh file to label
**** FILE INPUT ****** BHDATA OPTION=LOAD LABEL=outermumetal_door_BH FILE='bhdata//miniMSR_outer_door_anh.bh'
BHDATA label outermumetal_door_BH loaded with data from the file bhdata//miniMSR_outer_door_anh.bh
**** FILE INPUT ****** BHDATA OPTION=LOAD LABEL=outermumetal_floor_BH FILE='bhdata//miniMSR_outer_floor_anh.bh'
BHDATA label outermumetal_floor_BH loaded with data from the file bhdata//miniMSR_outer_floor_anh.bh
**** FILE INPUT ******
*** COMMENT ********** / setup boundary condition
**** FILE INPUT ****** BOUNDARY UNPICK
**** FILE INPUT ****** BOUNDARY GUIINIT
**** FILE INPUT ****** BOUNDARY PICK 'outer_mumetal_door'
**** FILE INPUT ****** BOUNDARY OPTION=MODIFY CONDITION=THINPLATE PLATEMATERIAL=outer_mumetal_door PLATETHICKNESS=0.002
**** FILE INPUT ****** BOUNDARY UNPICK
**** FILE INPUT ****** BOUNDARY UNPICK
**** FILE INPUT ****** BOUNDARY GUIINIT
**** FILE INPUT ****** BOUNDARY PICK 'outer_mumetal_floor'
**** FILE INPUT ****** BOUNDARY OPTION=MODIFY CONDITION=THINPLATE PLATEMATERIAL=outer_mumetal_floor PLATETHICKNESS=0.002
**** FILE INPUT ****** BOUNDARY UNPICK
**** FILE INPUT ******
*** COMMENT ********** / match material label to bh label
**** FILE INPUT ****** MATERIALS UNPICK
**** FILE INPUT ****** MATERIALS GUIINIT
**** FILE INPUT ****** MATERIALS PICK 'outer_mumetal_door'
**** FILE INPUT ****** MATERIALS OPTION=MODIFY MULINEARITY=NONLINEAR MUANISOTROPY=ISOTROPIC BH='outermumetal_door_BH'
**** FILE INPUT ****** MATERIALS UNPICK
**** FILE INPUT ****** MATERIALS UNPICK
**** FILE INPUT ****** MATERIALS GUIINIT
**** FILE INPUT ****** MATERIALS PICK 'outer_mumetal_floor'
**** FILE INPUT ****** MATERIALS OPTION=MODIFY MULINEARITY=NONLINEAR MUANISOTROPY=ISOTROPIC BH='outermumetal_floor_BH'
**** FILE INPUT ****** MATERIALS UNPICK
**** FILE INPUT ******
**** FILE INPUT ******
*** COMMENT ********** // create total air block
**** FILE INPUT ****** SKETCH OBJECT=BLOCK +COMPLETE
**** FILE INPUT ****** PREVIEW OPTION=END REDISPLAY=NO
**** FILE INPUT ****** BLOCK Name='Block' X0=0.5 Y0=0.5 Z0=0.5 X1=-0.5 Y1=-0.5 Z1=-0.5 MATERIALLABEL='Air' LEVEL=#air_tot_DSL
*** COMMENT ********** / rename
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Block'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='total_air_block' UNIQUENAME='total_air_block'
*** COMMENT ********** / mesh parameters
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='total_air_block'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=#air_tot_DSL SIZE=#air_tot_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
**** FILE INPUT ******
*** COMMENT ********** // create reduced air block
**** FILE INPUT ****** SKETCH OBJECT=BLOCK +COMPLETE
**** FILE INPUT ****** PREVIEW OPTION=END REDISPLAY=NO
**** FILE INPUT ****** BLOCK Name='Block' X0=2 Y0=2 Z0=2 X1=-2 Y1=-2 Z1=-2 MATERIALLABEL='Air' LEVEL=#air_red_DSL
*** COMMENT ********** / rename
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Block'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='reduced_air_block' UNIQUENAME='reduced_air_block'
*** COMMENT ********** / mesh parameters
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='reduced_air_block'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Reduced ELEMENTTYPE=Linear LEVEL=#air_red_DSL SIZE=#air_red_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
**** FILE INPUT ******
*** COMMENT ********** / zoom out view
**** FILE INPUT ****** THREED XORIGIN=0 YORIGIN=0 ZORIGIN=0 ROTX=20 ROTY=45 ROTZ=0 XASPECT=1 YASPECT=1 ZASPECT=1 SIZE=5 FACETANGLE=5 PERSPECTIVE=YES LINECOLOUR=YES OPTION=SETVIEW
**** FILE INPUT ******
*** COMMENT ********** //// Analysis Settings
**** FILE INPUT ****** MULTIPHYSICS OPTION=RESET
**** FILE INPUT ****** ANALYSISDATA OPTION=SET PROGRAM=TOSCAMAGN HX=0 HY=0 HZ=50 LINEAR=NO NITERATIONS=21 NLITERTYPE=NEWTON POTENTIALCUT=YES RHS=ADAPTIVE SCALEDRIVE=ALL TOLERANCE=0.001 USEDEFORMEDMESH=NO USEDIRECTSOLVER=NO
**** FILE INPUT ******
*** COMMENT ********** //// Symmetry
*** COMMENT ********** /++ BACKGROUND OPTION=LOAD
*** COMMENT ********** /++ BACKGROUND OPTION=SET SHAPE=BLOCK SCALEX=1 SCALEY=1 SCALEZ=1   XYSYMMETRYPLANE=NO YZSYMMETRYPLANE=YES ROTZNUM=1 ZXSYMMETRYPLANE=YES EMRYZ=TANGMAGN EMRZX=NORMMAGN
**** FILE INPUT ******
**** FILE INPUT ******
*** COMMENT ********** //// Create Model Body
**** FILE INPUT ****** MODEL CREATE
**** FILE INPUT ******
*** COMMENT ********** //// Meshing
*** COMMENT ********** / Surface Mesh
**** FILE INPUT ****** MESH GENERATOR=AUTOMATIC SIZE=0.3 NORMALTOL=30.0 SURFACETOL=0.0 TOLERANCE=1.0E-06 TYPE=PREFERTETRA
Mesh generator Type II
696 faces completed, generating 240586 surface elements

Triangular facet quality on 696 faces
Triangles: 240586, Max: 1.0, Min: 0.159128, Mean: 0.928738, Power2Mean: 0.818231

**** FILE INPUT ******
*** COMMENT ********** / Volume Mesh
**** FILE INPUT ****** FILL TOL=1.0E-06

Tetrahedral element quality on 38 cells
Tetrahedra: 1138992, Max: 1.631634, Min: 0.0850922, RMS: 1.017377, Power2Mean: 0.874831, Worst cell: 34

Total nodes:    193586
Tetrahedra:     1138992
Total volume: 64.0
**** FILE INPUT ******
*** COMMENT ********** //// Create Database
**** FILE INPUT ****** SOLVERS SOLVENOW=NO SAVEMODEL=YES,
**** FILE INPUT ****** SOLVERS OPTION=TEST FILE='mini_MSR' UNITS=METRE ELEMENT=MIXED SURFACE=CURVED
**** FILE INPUT ****** COMMENT CLEAR=YES TYPE=DBTITLE
**** FILE INPUT ****** SOLVERS OPTION=OVERWRITE
Opening database mini_MSR.op3, simulation number 0 on 23/May/2024 at 11:02:17
   Magnetostatic Analysis (TOSCA)

   File: mini_MSR.op3 simulation:  1
   Created on: 23/May/2024 11:02:02
   In Directory:
   C:\Users\mzhao\OPERA_miniMSR\
   By Machine:
   Node: UCNENG01. Processor: Intel64 Family 6 Model 42 Stepping 7, GenuineIntel. System: Windows 10 Version 2009. version windows (10.0.19045)
   Log Files: Modeller_202405230944315521.log/lp
   Simulation created using: Opera-3d/Modeller Version 21.3.3
   User did not enter a title

   SI units

   No conductors
   External field vector (0.0,0.0,50.0 A/m)
   Adaptive RHS integrals

Warning: Automatic potential cuts have been disabled because there are thin plates.

   Boundary conditions:
     TANGMAGN
        Note: Assumed as the default boundary condition to one or more external faces.
     THINPLATE
        thin plate of material OUTER_MUMETAL_FLOOR, thickness: 2.0E-03 [m]
        thin plate of material OUTER_MUMETAL_DOOR, thickness: 2.0E-03 [m]
        thin plate of material INNER_MUMETAL_FLOOR, thickness: 1.5E-03 [m]
        thin plate of material INNER_MUMETAL_DOOR, thickness: 1.5E-03 [m]

   Nonlinear iteration data
   Newton-Raphson Iterations
   Maximum Iterations: 21
   Tolerance: 1.0E-03

   No periodicity conditions have been defined

   1 solution scaling factor defined for all drives: 1.0

   Materials defined:
   OUTER_MUMETAL_FLOOR
     Isotropic permeability: bhdata//miniMSR_outer_floor_anh.bh (15193.9 )
   OUTER_MUMETAL_DOOR
     Isotropic permeability: bhdata//miniMSR_outer_door_anh.bh (5933.308)
   INNER_MUMETAL_FLOOR
     Isotropic permeability: bhdata//miniMSR_inner_floor_anh.bh (25009.79)
   INNER_MUMETAL_DOOR
     Isotropic permeability: 1.0
   AIR

   808813 nodes in the model
   1694790 edges in the model
   Both linear and quadratic elements exist in the model
   1138992 tetrahedral elements (878837 linear and 260155 quadratic)
   1138992 elements in total

   Symmetry:
   none

WARNINGS:
1 warning when preparing simulation
**** FILE INPUT ******
**** FILE INPUT ******
**** FILE INPUT ******
**** FILE INPUT ******
**** FILE INPUT ******
End of command file.
*** CONTROL COMMAND ** $ END COMI
**** COMMAND INPUT ***
**** COMMAND INPUT *** BHDATA OPTION=VIEW  label=innermumetal_door_BH
**** COMMAND INPUT ***
*** CONTROL COMMAND ** $ COMI 'C:\Users\mzhao\OPERA_miniMSR\test.comi'
Opening file for input: C:\Users\mzhao\OPERA_miniMSR\test.comi
*** COMMENT ********** // this script creates model of mini-MSR in OPERA
*** COMMENT ********** // author: M. Zhao
*** COMMENT ********** // May 2024
*** CONTROL COMMAND ** $string yesorno YES
 yesorno='YES'
**** FILE INPUT ****** CLEAR REVERT=NO
Data changes within the model will be lost.
(this operation cannot be UNDONE)
Continue?
The default value has been used from the YESORNO string variable (YES)
Command language initialization.
**** FILE INPUT ****** THREED XORIGIN=0 YORIGIN=0 ZORIGIN=0 ROTX=0 ROTY=0 ROTZ=0 XASPECT=1 YASPECT=1 ZASPECT=1 SIZE=0 FACETANGLE=5 PERSPECTIVE=YES LINECOLOUR=YES OPTION=SETVIEW
**** FILE INPUT ******
*** COMMENT ********** ////////////////////////////////////////////////////////////
*** COMMENT ********** / Constants
*** COMMENT ********** ////////////////////////////////////////////////////////////
*** COMMENT ********** //Data Storage Levels
*** COMMENT ********** / mumetal
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#mu_DSL, Value=500, DESCRIPTION='data storage level for mumetal'
Variable      Value            Expression
#mu_DSL       500.0            500
Description: data storage level for mumetal
*** COMMENT ********** / air (total potential)
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#air_tot_DSL, Value=200, DESCRIPTION='data storage level for total potential air'
Variable      Value            Expression
#air_tot_DSL  200.0            200
Description: data storage level for total potential air
*** COMMENT ********** / air (reduced potential)
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#air_red_DSL, Value=100, DESCRIPTION='data storage level for reduced potential air'
Variable      Value            Expression
#air_red_DSL  100.0            100
Description: data storage level for reduced potential air
**** FILE INPUT ******
*** COMMENT ********** /// meshing parameters
*** COMMENT ********** // max element size (MES)
*** COMMENT ********** / flat mumetal
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#mu_flat_MES, Value=0.05, DESCRIPTION='maximum element size for flat mumetal'
Variable      Value            Expression
#mu_flat_MES  0.05             0.05
Description: maximum element size for flat mumetal
*** COMMENT ********** / bent mumetal
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#mu_bent_MES, Value=0.03, DESCRIPTION='maximum element size for bent mumetal'
Variable      Value            Expression
#mu_bent_MES  0.03             0.03
Description: maximum element size for bent mumetal
*** COMMENT ********** / air (total potential)
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#air_tot_MES, Value=0.02, DESCRIPTION='maximum element size for total potential air'
Variable      Value            Expression
#air_tot_MES  0.02             0.02
Description: maximum element size for total potential air
*** COMMENT ********** / air (reduced potential)
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#air_red_MES, Value=0.3, DESCRIPTION='maximum element size for reduced potential air'
Variable      Value            Expression
#air_red_MES  0.3              0.3
Description: maximum element size for reduced potential air
**** FILE INPUT ******
*** COMMENT ********** // maximum angle between elements (MAbE)
*** COMMENT ********** / flat mumetal
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#mu_flat_MAbE, Value=30, DESCRIPTION='maximum angle between elements for flat mumetal'
Variable      Value            Expression
#mu_flat_MAbE 30.0             30
Description: maximum angle between elements for flat mumetal
*** COMMENT ********** / bent mumetal
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#mu_bent_MAbE, Value=30, DESCRIPTION='maximum angle between elements for bent mumetal'
Variable      Value            Expression
#mu_bent_MAbE 30.0             30
Description: maximum angle between elements for bent mumetal
*** COMMENT ********** / air
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#air_MAbE, Value=30, DESCRIPTION='maximum angle between elements for air'
Variable      Value            Expression
#air_MAbE     30.0             30
Description: maximum angle between elements for air
**** FILE INPUT ******
**** FILE INPUT ******
*** COMMENT ********** ////////////////////////////////////////////////////////////
*** COMMENT ********** / Modelling
*** COMMENT ********** ////////////////////////////////////////////////////////////
**** FILE INPUT ******
*** COMMENT ********** //// Inner Layer
**** FILE INPUT ******
*** COMMENT ********** / load structural step file
**** FILE INPUT ****** LOAD OPTION=INSERT FILE='stpfiles//inner_layer.STEP'
Imported material name from label: layer1_NAUO1#inner_flat_front_Boss_Extrude2
Imported material name from label: layer1_NAUO2#inner_bracket_Cut_Extrude3
Imported material name from label: layer1_NAUO3#inner_bracket_Cut_Extrude3
Imported material name from label: layer1_NAUO4#inner_bracket_Cut_Extrude3
Imported material name from label: layer1_NAUO5#inner_bracket_Cut_Extrude3
Imported material name from label: layer1_NAUO6#inner_flat_Boss_Extrude2
Imported material name from label: layer1_NAUO7#inner_flat_Boss_Extrude2
Imported material name from label: layer1_NAUO8#inner_bracket_Cut_Extrude3
Imported material name from label: layer1_NAUO9#inner_bracket_Cut_Extrude3
Imported material name from label: layer1_NAUO10#inner_flat_Boss_Extrude2
Imported material name from label: layer1_NAUO11#inner_bracket_Cut_Extrude3
Imported material name from label: layer1_NAUO12#inner_bracket_Cut_Extrude3
Imported material name from label: layer1_NAUO13#inner_bracket_Cut_Extrude3
Imported material name from label: layer1_NAUO14#inner_bracket_Cut_Extrude3
Imported material name from label: layer1_NAUO15#inner_bracket_Cut_Extrude3
Imported material name from label: layer1_NAUO16#inner_bracket_Cut_Extrude3
Imported material name from label: layer1_NAUO17#inner_flat_Boss_Extrude2
Imported material name from label: layer1_NAUO18#inner_flat_Boss_Extrude2
**** FILE INPUT ******
*** COMMENT ********** /// Flat panels
*** COMMENT ********** // front
*** COMMENT ********** / rename body
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-001'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_front' UNIQUENAME='inner_front'
*** COMMENT ********** / set material to air
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_front'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=#air_tot_DSL SIZE=#air_tot_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / face settings
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** / set face to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_front' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_flat_MES NORMALTOL=#mu_flat_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
*** COMMENT ********** // back
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-018'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_back' UNIQUENAME='inner_back'
*** COMMENT ********** / set material to air
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_back'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=#air_tot_DSL SIZE=#air_tot_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / face settings
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** / set face to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_back' IDENTIFIER=A.00010
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_flat_MES NORMALTOL=#mu_flat_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
*** COMMENT ********** // left
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-010'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_left' UNIQUENAME='inner_left'
*** COMMENT ********** / set material to air
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_left'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=#air_tot_DSL SIZE=#air_tot_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / face settings
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** / set face to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_left' IDENTIFIER=A.00010
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_flat_MES NORMALTOL=#mu_flat_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
*** COMMENT ********** // right
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-006'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_right' UNIQUENAME='inner_right'
*** COMMENT ********** / set material to air
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_right'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=#air_tot_DSL SIZE=#air_tot_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / face settings
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** / set face to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_right' IDENTIFIER=A.00010
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_flat_MES NORMALTOL=#mu_flat_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
*** COMMENT ********** // top
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-017'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_top' UNIQUENAME='inner_top'
*** COMMENT ********** / set material to air
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_top'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=#air_tot_DSL SIZE=#air_tot_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / face settings
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** / set face to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_top' IDENTIFIER=A.00010
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_floor' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_flat_MES NORMALTOL=#mu_flat_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
*** COMMENT ********** // bottom
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-007'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_bottom' UNIQUENAME='inner_bottom'
*** COMMENT ********** / set material to air
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_bottom'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=#air_tot_DSL SIZE=#air_tot_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / face settings
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** / set face to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_bottom' IDENTIFIER=A.00010
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_floor' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_flat_MES NORMALTOL=#mu_flat_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ******
*** COMMENT ********** /// brackets
*** COMMENT ********** // front-top
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-005'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_brkt_front_top' UNIQUENAME='inner_brkt_front_top'
*** COMMENT ********** / set material to air
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_brkt_front_top'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=#air_tot_DSL SIZE=#air_tot_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / face settings
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** / set face to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_front_top' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_front_top' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_front_top' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
*** COMMENT ********** // front-bottom
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-003'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_brkt_front_bottom' UNIQUENAME='inner_brkt_front_bottom'
*** COMMENT ********** / set material to air
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_brkt_front_bottom'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=#air_tot_DSL SIZE=#air_tot_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / face settings
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** / set face to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_front_bottom' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_front_bottom' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_front_bottom' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
*** COMMENT ********** // front-left
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-004'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_brkt_front_left' UNIQUENAME='inner_brkt_front_left'
*** COMMENT ********** / set material to air
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_brkt_front_left'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=#air_tot_DSL SIZE=#air_tot_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / face settings
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** / set face to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_front_left' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_front_left' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_front_left' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
*** COMMENT ********** // front-right
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-002'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_brkt_front_right' UNIQUENAME='inner_brkt_front_right'
*** COMMENT ********** / set material to air
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_brkt_front_right'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=#air_tot_DSL SIZE=#air_tot_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / face settings
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** / set face to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_front_right' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_front_right' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_front_right' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
*** COMMENT ********** // bottom-left
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-012'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_brkt_bottom-left' UNIQUENAME='inner_brkt_bottom-left'
*** COMMENT ********** / set material to air
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_brkt_bottom-left'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=#air_tot_DSL SIZE=#air_tot_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / face settings
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** / set face to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_bottom-left' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_floor' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_bottom-left' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_floor' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_bottom-left' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_floor' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
*** COMMENT ********** // bottom-right
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-008'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_brkt_bottom_right' UNIQUENAME='inner_brkt_bottom_right'
*** COMMENT ********** / set material to air
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_brkt_bottom_right'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=#air_tot_DSL SIZE=#air_tot_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / face settings
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** / set face to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_bottom_right' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_floor' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_bottom_right' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_floor' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_bottom_right' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_floor' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
*** COMMENT ********** // top-left
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-011'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_brkt_top_left' UNIQUENAME='inner_brkt_top_left'
*** COMMENT ********** / set material to air
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_brkt_top_left'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=#air_tot_DSL SIZE=#air_tot_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / face settings
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** / set face to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_top_left' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_floor' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_top_left' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_floor' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_top_left' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_floor' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
*** COMMENT ********** // top-right
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-009'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_brkt_top_right' UNIQUENAME='inner_brkt_top_right'
*** COMMENT ********** / set material to air
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_brkt_top_right'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=#air_tot_DSL SIZE=#air_tot_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / face settings
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** / set face to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_top_right' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_floor' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_top_right' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_floor' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_top_right' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_floor' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
*** COMMENT ********** // back-bottom
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-013'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_brkt_back_bottom' UNIQUENAME='inner_brkt_back_bottom'
*** COMMENT ********** / set material to air
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_brkt_back_bottom'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=#air_tot_DSL SIZE=#air_tot_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / face settings
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** / set face to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_back_bottom' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_back_bottom' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_back_bottom' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
*** COMMENT ********** // back-top
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-015'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_brkt_back_top' UNIQUENAME='inner_brkt_back_top'
*** COMMENT ********** / set material to air
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_brkt_back_top'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=#air_tot_DSL SIZE=#air_tot_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / face settings
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** / set face to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_back_top' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_back_top' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_back_top' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
*** COMMENT ********** // back-left
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-016'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_brkt_back_left' UNIQUENAME='inner_brkt_back_left'
*** COMMENT ********** / set material to air
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_brkt_back_left'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=#air_tot_DSL SIZE=#air_tot_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / face settings
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** / set face to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_back_left' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_back_left' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_back_left' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
*** COMMENT ********** // back-right
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-014'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_brkt_back_right' UNIQUENAME='inner_brkt_back_right'
*** COMMENT ********** / set material to air
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_brkt_back_right'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=#air_tot_DSL SIZE=#air_tot_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / face settings
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** / set face to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_back_right' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_back_right' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_back_right' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ******
**** FILE INPUT ******
*** COMMENT ********** // BH data
*** COMMENT ********** / create labels
**** FILE INPUT ****** BHDATA OPTION=NEW  LABEL='innermumetal_door_BH'
**** FILE INPUT ****** BHDATA OPTION=NEW  LABEL='innermumetal_floor_BH'
**** FILE INPUT ******
*** COMMENT ********** / assign bh file to label
**** FILE INPUT ****** BHDATA OPTION=LOAD LABEL=innermumetal_door_BH FILE='bhdata//miniMSR_inner_door_anh.bh'
BHDATA label innermumetal_door_BH loaded with data from the file bhdata//miniMSR_inner_door_anh.bh
**** FILE INPUT ****** BHDATA OPTION=LOAD LABEL=innermumetal_floor_BH FILE='bhdata//miniMSR_inner_floor_anh.bh'
BHDATA label innermumetal_floor_BH loaded with data from the file bhdata//miniMSR_inner_floor_anh.bh
**** FILE INPUT ******
*** COMMENT ********** / setup boundary condition
**** FILE INPUT ****** BOUNDARY UNPICK
**** FILE INPUT ****** BOUNDARY GUIINIT
**** FILE INPUT ****** BOUNDARY PICK 'inner_mumetal_door'
**** FILE INPUT ****** BOUNDARY OPTION=MODIFY CONDITION=THINPLATE PLATEMATERIAL=inner_mumetal_door PLATETHICKNESS=0.0015
**** FILE INPUT ****** BOUNDARY UNPICK
**** FILE INPUT ******
**** FILE INPUT ****** BOUNDARY UNPICK
**** FILE INPUT ****** BOUNDARY GUIINIT
**** FILE INPUT ****** BOUNDARY PICK 'inner_mumetal_floor'
**** FILE INPUT ****** BOUNDARY OPTION=MODIFY CONDITION=THINPLATE PLATEMATERIAL=inner_mumetal_floor PLATETHICKNESS=0.0015
**** FILE INPUT ****** BOUNDARY UNPICK
**** FILE INPUT ******
*** COMMENT ********** / match material label to bh label
**** FILE INPUT ****** MATERIALS UNPICK
**** FILE INPUT ****** MATERIALS GUIINIT
**** FILE INPUT ****** MATERIALS PICK 'inner_mumetal_floor'
**** FILE INPUT ****** MATERIALS OPTION=MODIFY MULINEARITY=NONLINEAR MUANISOTROPY=ISOTROPIC BH='innermumetal_door_BH'
**** FILE INPUT ****** MATERIALS UNPICK
**** FILE INPUT ****** MATERIALS UNPICK
**** FILE INPUT ****** MATERIALS GUIINIT
**** FILE INPUT ****** MATERIALS PICK 'inner_mumetal_floor'
**** FILE INPUT ****** MATERIALS OPTION=MODIFY MULINEARITY=NONLINEAR MUANISOTROPY=ISOTROPIC BH='innermumetal_floor_BH'
**** FILE INPUT ****** MATERIALS UNPICK
**** FILE INPUT ******
**** FILE INPUT ******
*** COMMENT ********** //// Outer Layer
**** FILE INPUT ******
*** COMMENT ********** / load structural step file
**** FILE INPUT ****** LOAD OPTION=INSERT FILE='stpfiles//outer_layer.STEP'
Imported material name from label: outer_layer_NAUO1#outer_flat_front_Boss_Extrude2
Imported material name from label: outer_layer_NAUO2#outer_bracket_Cut_Extrude4
Imported material name from label: outer_layer_NAUO3#outer_bracket_Cut_Extrude4
Imported material name from label: outer_layer_NAUO4#outer_flat_Boss_Extrude2
Imported material name from label: outer_layer_NAUO5#outer_flat_Boss_Extrude2
Imported material name from label: outer_layer_NAUO6#outer_bracket_Cut_Extrude4
Imported material name from label: outer_layer_NAUO7#outer_bracket_Cut_Extrude4
Imported material name from label: outer_layer_NAUO8#outer_bracket_Cut_Extrude4
Imported material name from label: outer_layer_NAUO9#outer_bracket_Cut_Extrude4
Imported material name from label: outer_layer_NAUO10#outer_flat_Boss_Extrude2
Imported material name from label: outer_layer_NAUO11#outer_bracket_Cut_Extrude4
Imported material name from label: outer_layer_NAUO12#outer_bracket_Cut_Extrude4
Imported material name from label: outer_layer_NAUO13#outer_bracket_Cut_Extrude4
Imported material name from label: outer_layer_NAUO14#outer_flat_Boss_Extrude2
Imported material name from label: outer_layer_NAUO15#outer_bracket_Cut_Extrude4
Imported material name from label: outer_layer_NAUO16#outer_bracket_Cut_Extrude4
Imported material name from label: outer_layer_NAUO17#outer_bracket_Cut_Extrude4
Imported material name from label: outer_layer_NAUO18#outer_flat_Boss_Extrude2
Imported material name from label: outer_layer_NAUO19#outer_bracket_Cut_Extrude4
Imported material name from label: outer_layer_NAUO20#outer_bracket_Cut_Extrude4
Imported material name from label: outer_layer_NAUO21#outer_bracket_Cut_Extrude4
Imported material name from label: outer_layer_NAUO22#outer_bracket_Cut_Extrude4
Imported material name from label: outer_layer_NAUO23#outer_bracket_Cut_Extrude4
Imported material name from label: outer_layer_NAUO24#outer_bracket_Cut_Extrude4
Imported material name from label: outer_layer_NAUO25#outer_flat_Boss_Extrude2
Imported material name from label: outer_layer_NAUO26#outer_flat_Boss_Extrude2
**** FILE INPUT ******
*** COMMENT ********** /// Flat panels
*** COMMENT ********** // front
*** COMMENT ********** / rename body
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-001'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='outer_front' UNIQUENAME='outer_front'
*** COMMENT ********** / set material to air
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_front'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=#air_tot_DSL SIZE=#air_tot_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / face settings
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** / set face to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_front' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_flat_MES NORMALTOL=#mu_flat_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
*** COMMENT ********** // back
*** COMMENT ********** / rename body
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-026'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='outer_back' UNIQUENAME='outer_back'
*** COMMENT ********** / set material to air
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_back'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=#air_tot_DSL SIZE=#air_tot_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / face settings
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** / set face to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_back' IDENTIFIER=A.00010
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_flat_MES NORMALTOL=#mu_flat_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
*** COMMENT ********** // left
*** COMMENT ********** / rename body
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-014'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='outer_left' UNIQUENAME='outer_left'
*** COMMENT ********** / set material to air
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_left'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=#air_tot_DSL SIZE=#air_tot_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / face settings
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** / set face to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_left' IDENTIFIER=A.00010
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_flat_MES NORMALTOL=#mu_flat_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
*** COMMENT ********** // right
*** COMMENT ********** / rename body
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-018'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='outer_right' UNIQUENAME='outer_right'
*** COMMENT ********** / set material to air
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_right'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=#air_tot_DSL SIZE=#air_tot_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / face settings
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** / set face to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_right' IDENTIFIER=A.00010
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_flat_MES NORMALTOL=#mu_flat_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
*** COMMENT ********** // top
*** COMMENT ********** / rename body
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-025'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='outer_top' UNIQUENAME='outer_top'
*** COMMENT ********** / set material to air
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_top'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=#air_tot_DSL SIZE=#air_tot_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / face settings
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** / set face to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_top' IDENTIFIER=A.00010
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_floor' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_flat_MES NORMALTOL=#mu_flat_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
*** COMMENT ********** // bottom
*** COMMENT ********** / rename body
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-005'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='outer_bottom' UNIQUENAME='outer_bottom'
*** COMMENT ********** / set material to air
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_bottom'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=#air_tot_DSL SIZE=#air_tot_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / face settings
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** / set face to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_bottom' IDENTIFIER=A.00010
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_floor' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_flat_MES NORMALTOL=#mu_flat_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
*** COMMENT ********** /// brackets
*** COMMENT ********** // front-top
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-021'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='outer_brkt_front_top' UNIQUENAME='outer_brkt_front_top'
*** COMMENT ********** / set material to air
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_brkt_front_top'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=#air_tot_DSL SIZE=#air_tot_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / face settings
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** / set face to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_front_top' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_front_top' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_front_top' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
*** COMMENT ********** // front-bottom
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-003'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='outer_brkt_front_bottom' UNIQUENAME='outer_brkt_front_bottom'
*** COMMENT ********** / set material to air
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_brkt_front_bottom'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=#air_tot_DSL SIZE=#air_tot_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / face settings
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** / set face to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_front_bottom' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_front_bottom' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_front_bottom' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
*** COMMENT ********** // front-left
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-013'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='outer_brkt_front_left' UNIQUENAME='outer_brkt_front_left'
*** COMMENT ********** / set material to air
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_brkt_front_left'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=#air_tot_DSL SIZE=#air_tot_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / face settings
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** / set face to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_front_left' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_front_left' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_front_left' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
*** COMMENT ********** // front-right
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-017'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='outer_brkt_front_right' UNIQUENAME='outer_brkt_front_right'
*** COMMENT ********** / set material to air
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_brkt_front_right'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=#air_tot_DSL SIZE=#air_tot_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / face settings
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** / set face to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_front_right' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_front_right' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_front_right' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
*** COMMENT ********** // bottom-left
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-015'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='outer_brkt_bottom-left' UNIQUENAME='outer_brkt_bottom-left'
*** COMMENT ********** / set material to air
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_brkt_bottom-left'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=#air_tot_DSL SIZE=#air_tot_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / face settings
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** / set face to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_bottom-left' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_floor' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_bottom-left' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_floor' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_bottom-left' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_floor' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
*** COMMENT ********** // bottom-right
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-019'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='outer_brkt_bottom_right' UNIQUENAME='outer_brkt_bottom_right'
*** COMMENT ********** / set material to air
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_brkt_bottom_right'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=#air_tot_DSL SIZE=#air_tot_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / face settings
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** / set face to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_bottom_right' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_floor' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_bottom_right' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_floor' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_bottom_right' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_floor' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
*** COMMENT ********** // top-left
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-022'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='outer_brkt_top_left' UNIQUENAME='outer_brkt_top_left'
*** COMMENT ********** / set material to air
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_brkt_top_left'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=#air_tot_DSL SIZE=#air_tot_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / face settings
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** / set face to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_top_left' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_floor' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_top_left' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_floor' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_top_left' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_floor' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
*** COMMENT ********** // top-right
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-024'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='outer_brkt_top_right' UNIQUENAME='outer_brkt_top_right'
*** COMMENT ********** / set material to air
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_brkt_top_right'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=#air_tot_DSL SIZE=#air_tot_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / face settings
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** / set face to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_top_right' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_floor' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_top_right' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_floor' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_top_right' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_floor' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
*** COMMENT ********** // back-bottom
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-020'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='outer_brkt_back_bottom' UNIQUENAME='outer_brkt_back_bottom'
*** COMMENT ********** / set material to air
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_brkt_back_bottom'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=#air_tot_DSL SIZE=#air_tot_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / face settings
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** / set face to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_back_bottom' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_back_bottom' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_back_bottom' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
*** COMMENT ********** // back-top
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-023'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='outer_brkt_back_top' UNIQUENAME='outer_brkt_back_top'
*** COMMENT ********** / set material to air
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_brkt_back_top'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=#air_tot_DSL SIZE=#air_tot_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / face settings
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** / set face to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_back_top' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_back_top' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_back_top' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
*** COMMENT ********** // back-left
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-012'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='outer_brkt_back_left' UNIQUENAME='outer_brkt_back_left'
*** COMMENT ********** / set material to air
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_brkt_back_left'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=#air_tot_DSL SIZE=#air_tot_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / face settings
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** / set face to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_back_left' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_back_left' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_back_left' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
*** COMMENT ********** // back-right
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-016'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='outer_brkt_back_right' UNIQUENAME='outer_brkt_back_right'
*** COMMENT ********** / set material to air
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_brkt_back_right'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=#air_tot_DSL SIZE=#air_tot_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / face settings
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** / set face to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_back_right' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_back_right' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_back_right' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ******
*** COMMENT ********** // BH data
*** COMMENT ********** / create label
**** FILE INPUT ****** BHDATA OPTION=NEW  LABEL='outermumetal_door_BH'
**** FILE INPUT ****** BHDATA OPTION=NEW  LABEL='outermumetal_floor_BH'
**** FILE INPUT ******
*** COMMENT ********** / assign bh file to label
**** FILE INPUT ****** BHDATA OPTION=LOAD LABEL=outermumetal_door_BH FILE='bhdata//miniMSR_outer_door_anh.bh'
BHDATA label outermumetal_door_BH loaded with data from the file bhdata//miniMSR_outer_door_anh.bh
**** FILE INPUT ****** BHDATA OPTION=LOAD LABEL=outermumetal_floor_BH FILE='bhdata//miniMSR_outer_floor_anh.bh'
BHDATA label outermumetal_floor_BH loaded with data from the file bhdata//miniMSR_outer_floor_anh.bh
**** FILE INPUT ******
*** COMMENT ********** / setup boundary condition
**** FILE INPUT ****** BOUNDARY UNPICK
**** FILE INPUT ****** BOUNDARY GUIINIT
**** FILE INPUT ****** BOUNDARY PICK 'outer_mumetal_door'
**** FILE INPUT ****** BOUNDARY OPTION=MODIFY CONDITION=THINPLATE PLATEMATERIAL=outer_mumetal_door PLATETHICKNESS=0.002
**** FILE INPUT ****** BOUNDARY UNPICK
**** FILE INPUT ****** BOUNDARY UNPICK
**** FILE INPUT ****** BOUNDARY GUIINIT
**** FILE INPUT ****** BOUNDARY PICK 'outer_mumetal_floor'
**** FILE INPUT ****** BOUNDARY OPTION=MODIFY CONDITION=THINPLATE PLATEMATERIAL=outer_mumetal_floor PLATETHICKNESS=0.002
**** FILE INPUT ****** BOUNDARY UNPICK
**** FILE INPUT ******
*** COMMENT ********** / match material label to bh label
**** FILE INPUT ****** MATERIALS UNPICK
**** FILE INPUT ****** MATERIALS GUIINIT
**** FILE INPUT ****** MATERIALS PICK 'outer_mumetal_door'
**** FILE INPUT ****** MATERIALS OPTION=MODIFY MULINEARITY=NONLINEAR MUANISOTROPY=ISOTROPIC BH='outermumetal_door_BH'
**** FILE INPUT ****** MATERIALS UNPICK
**** FILE INPUT ****** MATERIALS UNPICK
**** FILE INPUT ****** MATERIALS GUIINIT
**** FILE INPUT ****** MATERIALS PICK 'outer_mumetal_floor'
**** FILE INPUT ****** MATERIALS OPTION=MODIFY MULINEARITY=NONLINEAR MUANISOTROPY=ISOTROPIC BH='outermumetal_floor_BH'
**** FILE INPUT ****** MATERIALS UNPICK
**** FILE INPUT ******
**** FILE INPUT ******
*** COMMENT ********** // create total air block
**** FILE INPUT ****** SKETCH OBJECT=BLOCK +COMPLETE
**** FILE INPUT ****** PREVIEW OPTION=END REDISPLAY=NO
**** FILE INPUT ****** BLOCK Name='Block' X0=0.5 Y0=0.5 Z0=0.5 X1=-0.5 Y1=-0.5 Z1=-0.5 MATERIALLABEL='Air' LEVEL=#air_tot_DSL
*** COMMENT ********** / rename
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Block'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='total_air_block' UNIQUENAME='total_air_block'
*** COMMENT ********** / mesh parameters
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='total_air_block'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=#air_tot_DSL SIZE=#air_tot_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
**** FILE INPUT ******
*** COMMENT ********** // create reduced air block
**** FILE INPUT ****** SKETCH OBJECT=BLOCK +COMPLETE
**** FILE INPUT ****** PREVIEW OPTION=END REDISPLAY=NO
**** FILE INPUT ****** BLOCK Name='Block' X0=2 Y0=2 Z0=2 X1=-2 Y1=-2 Z1=-2 MATERIALLABEL='Air' LEVEL=#air_red_DSL
*** COMMENT ********** / rename
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Block'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='reduced_air_block' UNIQUENAME='reduced_air_block'
*** COMMENT ********** / mesh parameters
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='reduced_air_block'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Reduced ELEMENTTYPE=Linear LEVEL=#air_red_DSL SIZE=#air_red_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
**** FILE INPUT ******
*** COMMENT ********** / zoom out view
**** FILE INPUT ****** THREED XORIGIN=0 YORIGIN=0 ZORIGIN=0 ROTX=20 ROTY=45 ROTZ=0 XASPECT=1 YASPECT=1 ZASPECT=1 SIZE=5 FACETANGLE=5 PERSPECTIVE=YES LINECOLOUR=YES OPTION=SETVIEW
**** FILE INPUT ******
*** COMMENT ********** //// Analysis Settings
**** FILE INPUT ****** MULTIPHYSICS OPTION=RESET
**** FILE INPUT ****** ANALYSISDATA OPTION=SET PROGRAM=TOSCAMAGN HX=0 HY=0 HZ=50 LINEAR=NO NITERATIONS=21 NLITERTYPE=NEWTON POTENTIALCUT=YES RHS=ADAPTIVE SCALEDRIVE=ALL TOLERANCE=0.001 USEDEFORMEDMESH=NO USEDIRECTSOLVER=NO
**** FILE INPUT ******
*** COMMENT ********** //// Symmetry
*** COMMENT ********** /++ BACKGROUND OPTION=LOAD
*** COMMENT ********** /++ BACKGROUND OPTION=SET SHAPE=BLOCK SCALEX=1 SCALEY=1 SCALEZ=1   XYSYMMETRYPLANE=NO YZSYMMETRYPLANE=YES ROTZNUM=1 ZXSYMMETRYPLANE=YES EMRYZ=TANGMAGN EMRZX=NORMMAGN
**** FILE INPUT ******
**** FILE INPUT ******
*** COMMENT ********** //// Create Model Body
**** FILE INPUT ****** MODEL CREATE
**** FILE INPUT ******
*** COMMENT ********** //// Meshing
*** COMMENT ********** / Surface Mesh
**** FILE INPUT ****** MESH GENERATOR=AUTOMATIC SIZE=0.3 NORMALTOL=30.0 SURFACETOL=0.0 TOLERANCE=1.0E-06 TYPE=PREFERTETRA
Mesh generator Type II
696 faces completed, generating 240790 surface elements

Triangular facet quality on 696 faces
Triangles: 240790, Max: 1.0, Min: 0.151653, Mean: 0.928827, Power2Mean: 0.818224

**** FILE INPUT ******
*** COMMENT ********** / Volume Mesh
**** FILE INPUT ****** FILL TOL=1.0E-06

Tetrahedral element quality on 38 cells
Tetrahedra: 1141251, Max: 1.612384, Min: 0.109007, RMS: 1.017553, Power2Mean: 0.875404, Worst cell: 20

Total nodes:    193994
Tetrahedra:     1141251
Total volume: 64.0
**** FILE INPUT ******
*** COMMENT ********** //// Create Database
**** FILE INPUT ****** SOLVERS SOLVENOW=NO SAVEMODEL=YES,
**** FILE INPUT ****** SOLVERS OPTION=TEST FILE='mini_MSR' UNITS=METRE ELEMENT=MIXED SURFACE=CURVED
File: mini_MSR.opc already exists.
Do you want to overwrite?
YES or NO
**** FILE INPUT ****** COMMENT CLEAR=YES TYPE=DBTITLE
DCOD Message  32:Too many fields found on line. Maximum   1. (FREFMT)
Command file processing aborted
Check the log file or console for details.
Error in file: 'C:\Users\mzhao\OPERA_miniMSR\test.comi' (Line 782)
**** COMMAND INPUT ***
*** CONTROL COMMAND ** $ COMI 'C:\Users\mzhao\OPERA_miniMSR\test.comi'
Opening file for input: C:\Users\mzhao\OPERA_miniMSR\test.comi
*** COMMENT ********** // this script creates model of mini-MSR in OPERA
*** COMMENT ********** // author: M. Zhao
*** COMMENT ********** // May 2024
*** CONTROL COMMAND ** $string yesorno YES
 yesorno='YES'
**** FILE INPUT ****** CLEAR REVERT=NO
Data changes within the model will be lost.
(this operation cannot be UNDONE)
Continue?
The default value has been used from the YESORNO string variable (YES)
Command language initialization.
**** FILE INPUT ****** THREED XORIGIN=0 YORIGIN=0 ZORIGIN=0 ROTX=0 ROTY=0 ROTZ=0 XASPECT=1 YASPECT=1 ZASPECT=1 SIZE=0 FACETANGLE=5 PERSPECTIVE=YES LINECOLOUR=YES OPTION=SETVIEW
**** FILE INPUT ******
*** COMMENT ********** ////////////////////////////////////////////////////////////
*** COMMENT ********** / Constants
*** COMMENT ********** ////////////////////////////////////////////////////////////
*** COMMENT ********** //Data Storage Levels
*** COMMENT ********** / mumetal
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#mu_DSL, Value=500, DESCRIPTION='data storage level for mumetal'
Variable      Value            Expression
#mu_DSL       500.0            500
Description: data storage level for mumetal
*** COMMENT ********** / air (total potential)
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#air_tot_DSL, Value=200, DESCRIPTION='data storage level for total potential air'
Variable      Value            Expression
#air_tot_DSL  200.0            200
Description: data storage level for total potential air
*** COMMENT ********** / air (reduced potential)
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#air_red_DSL, Value=100, DESCRIPTION='data storage level for reduced potential air'
Variable      Value            Expression
#air_red_DSL  100.0            100
Description: data storage level for reduced potential air
**** FILE INPUT ******
*** COMMENT ********** /// meshing parameters
*** COMMENT ********** // max element size (MES)
*** COMMENT ********** / flat mumetal
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#mu_flat_MES, Value=0.05, DESCRIPTION='maximum element size for flat mumetal'
Variable      Value            Expression
#mu_flat_MES  0.05             0.05
Description: maximum element size for flat mumetal
*** COMMENT ********** / bent mumetal
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#mu_bent_MES, Value=0.03, DESCRIPTION='maximum element size for bent mumetal'
Variable      Value            Expression
#mu_bent_MES  0.03             0.03
Description: maximum element size for bent mumetal
*** COMMENT ********** / air (total potential)
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#air_tot_MES, Value=0.02, DESCRIPTION='maximum element size for total potential air'
Variable      Value            Expression
#air_tot_MES  0.02             0.02
Description: maximum element size for total potential air
*** COMMENT ********** / air (reduced potential)
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#air_red_MES, Value=0.3, DESCRIPTION='maximum element size for reduced potential air'
Variable      Value            Expression
#air_red_MES  0.3              0.3
Description: maximum element size for reduced potential air
**** FILE INPUT ******
*** COMMENT ********** // maximum angle between elements (MAbE)
*** COMMENT ********** / flat mumetal
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#mu_flat_MAbE, Value=30, DESCRIPTION='maximum angle between elements for flat mumetal'
Variable      Value            Expression
#mu_flat_MAbE 30.0             30
Description: maximum angle between elements for flat mumetal
*** COMMENT ********** / bent mumetal
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#mu_bent_MAbE, Value=30, DESCRIPTION='maximum angle between elements for bent mumetal'
Variable      Value            Expression
#mu_bent_MAbE 30.0             30
Description: maximum angle between elements for bent mumetal
*** COMMENT ********** / air
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#air_MAbE, Value=30, DESCRIPTION='maximum angle between elements for air'
Variable      Value            Expression
#air_MAbE     30.0             30
Description: maximum angle between elements for air
**** FILE INPUT ******
**** FILE INPUT ******
*** COMMENT ********** ////////////////////////////////////////////////////////////
*** COMMENT ********** / Modelling
*** COMMENT ********** ////////////////////////////////////////////////////////////
**** FILE INPUT ******
*** COMMENT ********** //// Inner Layer
**** FILE INPUT ******
*** COMMENT ********** / load structural step file
**** FILE INPUT ****** LOAD OPTION=INSERT FILE='stpfiles//inner_layer.STEP'
Imported material name from label: layer1_NAUO1#inner_flat_front_Boss_Extrude2
Imported material name from label: layer1_NAUO2#inner_bracket_Cut_Extrude3
Imported material name from label: layer1_NAUO3#inner_bracket_Cut_Extrude3
Imported material name from label: layer1_NAUO4#inner_bracket_Cut_Extrude3
Imported material name from label: layer1_NAUO5#inner_bracket_Cut_Extrude3
Imported material name from label: layer1_NAUO6#inner_flat_Boss_Extrude2
Imported material name from label: layer1_NAUO7#inner_flat_Boss_Extrude2
Imported material name from label: layer1_NAUO8#inner_bracket_Cut_Extrude3
Imported material name from label: layer1_NAUO9#inner_bracket_Cut_Extrude3
Imported material name from label: layer1_NAUO10#inner_flat_Boss_Extrude2
Imported material name from label: layer1_NAUO11#inner_bracket_Cut_Extrude3
Imported material name from label: layer1_NAUO12#inner_bracket_Cut_Extrude3
Imported material name from label: layer1_NAUO13#inner_bracket_Cut_Extrude3
Imported material name from label: layer1_NAUO14#inner_bracket_Cut_Extrude3
Imported material name from label: layer1_NAUO15#inner_bracket_Cut_Extrude3
Imported material name from label: layer1_NAUO16#inner_bracket_Cut_Extrude3
Imported material name from label: layer1_NAUO17#inner_flat_Boss_Extrude2
Imported material name from label: layer1_NAUO18#inner_flat_Boss_Extrude2
**** FILE INPUT ******
*** COMMENT ********** /// Flat panels
*** COMMENT ********** // front
*** COMMENT ********** / rename body
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-001'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_front' UNIQUENAME='inner_front'
*** COMMENT ********** / set material to air
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_front'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=#air_tot_DSL SIZE=#air_tot_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / face settings
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** / set face to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_front' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_flat_MES NORMALTOL=#mu_flat_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
*** COMMENT ********** // back
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-018'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_back' UNIQUENAME='inner_back'
*** COMMENT ********** / set material to air
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_back'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=#air_tot_DSL SIZE=#air_tot_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / face settings
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** / set face to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_back' IDENTIFIER=A.00010
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_flat_MES NORMALTOL=#mu_flat_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
*** COMMENT ********** // left
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-010'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_left' UNIQUENAME='inner_left'
*** COMMENT ********** / set material to air
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_left'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=#air_tot_DSL SIZE=#air_tot_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / face settings
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** / set face to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_left' IDENTIFIER=A.00010
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_flat_MES NORMALTOL=#mu_flat_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
*** COMMENT ********** // right
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-006'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_right' UNIQUENAME='inner_right'
*** COMMENT ********** / set material to air
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_right'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=#air_tot_DSL SIZE=#air_tot_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / face settings
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** / set face to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_right' IDENTIFIER=A.00010
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_flat_MES NORMALTOL=#mu_flat_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
*** COMMENT ********** // top
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-017'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_top' UNIQUENAME='inner_top'
*** COMMENT ********** / set material to air
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_top'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=#air_tot_DSL SIZE=#air_tot_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / face settings
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** / set face to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_top' IDENTIFIER=A.00010
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_floor' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_flat_MES NORMALTOL=#mu_flat_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
*** COMMENT ********** // bottom
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-007'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_bottom' UNIQUENAME='inner_bottom'
*** COMMENT ********** / set material to air
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_bottom'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=#air_tot_DSL SIZE=#air_tot_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / face settings
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** / set face to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_bottom' IDENTIFIER=A.00010
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_floor' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_flat_MES NORMALTOL=#mu_flat_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ******
*** COMMENT ********** /// brackets
*** COMMENT ********** // front-top
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-005'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_brkt_front_top' UNIQUENAME='inner_brkt_front_top'
*** COMMENT ********** / set material to air
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_brkt_front_top'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=#air_tot_DSL SIZE=#air_tot_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / face settings
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** / set face to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_front_top' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_front_top' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_front_top' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
*** COMMENT ********** // front-bottom
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-003'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_brkt_front_bottom' UNIQUENAME='inner_brkt_front_bottom'
*** COMMENT ********** / set material to air
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_brkt_front_bottom'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=#air_tot_DSL SIZE=#air_tot_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / face settings
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** / set face to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_front_bottom' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_front_bottom' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_front_bottom' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
*** COMMENT ********** // front-left
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-004'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_brkt_front_left' UNIQUENAME='inner_brkt_front_left'
*** COMMENT ********** / set material to air
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_brkt_front_left'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=#air_tot_DSL SIZE=#air_tot_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / face settings
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** / set face to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_front_left' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_front_left' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_front_left' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
*** COMMENT ********** // front-right
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-002'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_brkt_front_right' UNIQUENAME='inner_brkt_front_right'
*** COMMENT ********** / set material to air
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_brkt_front_right'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=#air_tot_DSL SIZE=#air_tot_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / face settings
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** / set face to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_front_right' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_front_right' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_front_right' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
*** COMMENT ********** // bottom-left
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-012'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_brkt_bottom-left' UNIQUENAME='inner_brkt_bottom-left'
*** COMMENT ********** / set material to air
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_brkt_bottom-left'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=#air_tot_DSL SIZE=#air_tot_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / face settings
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** / set face to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_bottom-left' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_floor' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_bottom-left' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_floor' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_bottom-left' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_floor' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
*** COMMENT ********** // bottom-right
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-008'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_brkt_bottom_right' UNIQUENAME='inner_brkt_bottom_right'
*** COMMENT ********** / set material to air
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_brkt_bottom_right'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=#air_tot_DSL SIZE=#air_tot_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / face settings
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** / set face to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_bottom_right' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_floor' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_bottom_right' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_floor' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_bottom_right' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_floor' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
*** COMMENT ********** // top-left
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-011'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_brkt_top_left' UNIQUENAME='inner_brkt_top_left'
*** COMMENT ********** / set material to air
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_brkt_top_left'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=#air_tot_DSL SIZE=#air_tot_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / face settings
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** / set face to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_top_left' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_floor' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_top_left' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_floor' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_top_left' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_floor' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
*** COMMENT ********** // top-right
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-009'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_brkt_top_right' UNIQUENAME='inner_brkt_top_right'
*** COMMENT ********** / set material to air
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_brkt_top_right'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=#air_tot_DSL SIZE=#air_tot_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / face settings
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** / set face to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_top_right' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_floor' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_top_right' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_floor' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_top_right' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_floor' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
*** COMMENT ********** // back-bottom
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-013'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_brkt_back_bottom' UNIQUENAME='inner_brkt_back_bottom'
*** COMMENT ********** / set material to air
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_brkt_back_bottom'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=#air_tot_DSL SIZE=#air_tot_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / face settings
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** / set face to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_back_bottom' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_back_bottom' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_back_bottom' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
*** COMMENT ********** // back-top
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-015'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_brkt_back_top' UNIQUENAME='inner_brkt_back_top'
*** COMMENT ********** / set material to air
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_brkt_back_top'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=#air_tot_DSL SIZE=#air_tot_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / face settings
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** / set face to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_back_top' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_back_top' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_back_top' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
*** COMMENT ********** // back-left
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-016'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_brkt_back_left' UNIQUENAME='inner_brkt_back_left'
*** COMMENT ********** / set material to air
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_brkt_back_left'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=#air_tot_DSL SIZE=#air_tot_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / face settings
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** / set face to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_back_left' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_back_left' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_back_left' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
*** COMMENT ********** // back-right
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-014'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_brkt_back_right' UNIQUENAME='inner_brkt_back_right'
*** COMMENT ********** / set material to air
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_brkt_back_right'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=#air_tot_DSL SIZE=#air_tot_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / face settings
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** / set face to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_back_right' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_back_right' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_back_right' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ******
**** FILE INPUT ******
*** COMMENT ********** // BH data
*** COMMENT ********** / create labels
**** FILE INPUT ****** BHDATA OPTION=NEW  LABEL='innermumetal_door_BH'
**** FILE INPUT ****** BHDATA OPTION=NEW  LABEL='innermumetal_floor_BH'
**** FILE INPUT ******
*** COMMENT ********** / assign bh file to label
**** FILE INPUT ****** BHDATA OPTION=LOAD LABEL=innermumetal_door_BH FILE='bhdata//miniMSR_inner_door_anh.bh'
BHDATA label innermumetal_door_BH loaded with data from the file bhdata//miniMSR_inner_door_anh.bh
**** FILE INPUT ****** BHDATA OPTION=LOAD LABEL=innermumetal_floor_BH FILE='bhdata//miniMSR_inner_floor_anh.bh'
BHDATA label innermumetal_floor_BH loaded with data from the file bhdata//miniMSR_inner_floor_anh.bh
**** FILE INPUT ******
*** COMMENT ********** / setup boundary condition
**** FILE INPUT ****** BOUNDARY UNPICK
**** FILE INPUT ****** BOUNDARY GUIINIT
**** FILE INPUT ****** BOUNDARY PICK 'inner_mumetal_door'
**** FILE INPUT ****** BOUNDARY OPTION=MODIFY CONDITION=THINPLATE PLATEMATERIAL=inner_mumetal_door PLATETHICKNESS=0.0015
**** FILE INPUT ****** BOUNDARY UNPICK
**** FILE INPUT ******
**** FILE INPUT ****** BOUNDARY UNPICK
**** FILE INPUT ****** BOUNDARY GUIINIT
**** FILE INPUT ****** BOUNDARY PICK 'inner_mumetal_floor'
**** FILE INPUT ****** BOUNDARY OPTION=MODIFY CONDITION=THINPLATE PLATEMATERIAL=inner_mumetal_floor PLATETHICKNESS=0.0015
**** FILE INPUT ****** BOUNDARY UNPICK
**** FILE INPUT ******
*** COMMENT ********** / match material label to bh label
**** FILE INPUT ****** MATERIALS UNPICK
**** FILE INPUT ****** MATERIALS GUIINIT
**** FILE INPUT ****** MATERIALS PICK 'inner_mumetal_floor'
**** FILE INPUT ****** MATERIALS OPTION=MODIFY MULINEARITY=NONLINEAR MUANISOTROPY=ISOTROPIC BH='innermumetal_door_BH'
**** FILE INPUT ****** MATERIALS UNPICK
**** FILE INPUT ****** MATERIALS UNPICK
**** FILE INPUT ****** MATERIALS GUIINIT
**** FILE INPUT ****** MATERIALS PICK 'inner_mumetal_floor'
**** FILE INPUT ****** MATERIALS OPTION=MODIFY MULINEARITY=NONLINEAR MUANISOTROPY=ISOTROPIC BH='innermumetal_floor_BH'
**** FILE INPUT ****** MATERIALS UNPICK
**** FILE INPUT ******
**** FILE INPUT ******
*** COMMENT ********** //// Outer Layer
**** FILE INPUT ******
*** COMMENT ********** / load structural step file
**** FILE INPUT ****** LOAD OPTION=INSERT FILE='stpfiles//outer_layer.STEP'
Imported material name from label: outer_layer_NAUO1#outer_flat_front_Boss_Extrude2
Imported material name from label: outer_layer_NAUO2#outer_bracket_Cut_Extrude4
Imported material name from label: outer_layer_NAUO3#outer_bracket_Cut_Extrude4
Imported material name from label: outer_layer_NAUO4#outer_flat_Boss_Extrude2
Imported material name from label: outer_layer_NAUO5#outer_flat_Boss_Extrude2
Imported material name from label: outer_layer_NAUO6#outer_bracket_Cut_Extrude4
Imported material name from label: outer_layer_NAUO7#outer_bracket_Cut_Extrude4
Imported material name from label: outer_layer_NAUO8#outer_bracket_Cut_Extrude4
Imported material name from label: outer_layer_NAUO9#outer_bracket_Cut_Extrude4
Imported material name from label: outer_layer_NAUO10#outer_flat_Boss_Extrude2
Imported material name from label: outer_layer_NAUO11#outer_bracket_Cut_Extrude4
Imported material name from label: outer_layer_NAUO12#outer_bracket_Cut_Extrude4
Imported material name from label: outer_layer_NAUO13#outer_bracket_Cut_Extrude4
Imported material name from label: outer_layer_NAUO14#outer_flat_Boss_Extrude2
Imported material name from label: outer_layer_NAUO15#outer_bracket_Cut_Extrude4
Imported material name from label: outer_layer_NAUO16#outer_bracket_Cut_Extrude4
Imported material name from label: outer_layer_NAUO17#outer_bracket_Cut_Extrude4
Imported material name from label: outer_layer_NAUO18#outer_flat_Boss_Extrude2
Imported material name from label: outer_layer_NAUO19#outer_bracket_Cut_Extrude4
Imported material name from label: outer_layer_NAUO20#outer_bracket_Cut_Extrude4
Imported material name from label: outer_layer_NAUO21#outer_bracket_Cut_Extrude4
Imported material name from label: outer_layer_NAUO22#outer_bracket_Cut_Extrude4
Imported material name from label: outer_layer_NAUO23#outer_bracket_Cut_Extrude4
Imported material name from label: outer_layer_NAUO24#outer_bracket_Cut_Extrude4
Imported material name from label: outer_layer_NAUO25#outer_flat_Boss_Extrude2
Imported material name from label: outer_layer_NAUO26#outer_flat_Boss_Extrude2
**** FILE INPUT ******
*** COMMENT ********** /// Flat panels
*** COMMENT ********** // front
*** COMMENT ********** / rename body
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-001'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='outer_front' UNIQUENAME='outer_front'
*** COMMENT ********** / set material to air
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_front'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=#air_tot_DSL SIZE=#air_tot_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / face settings
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** / set face to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_front' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_flat_MES NORMALTOL=#mu_flat_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
*** COMMENT ********** // back
*** COMMENT ********** / rename body
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-026'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='outer_back' UNIQUENAME='outer_back'
*** COMMENT ********** / set material to air
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_back'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=#air_tot_DSL SIZE=#air_tot_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / face settings
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** / set face to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_back' IDENTIFIER=A.00010
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_flat_MES NORMALTOL=#mu_flat_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
*** COMMENT ********** // left
*** COMMENT ********** / rename body
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-014'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='outer_left' UNIQUENAME='outer_left'
*** COMMENT ********** / set material to air
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_left'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=#air_tot_DSL SIZE=#air_tot_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / face settings
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** / set face to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_left' IDENTIFIER=A.00010
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_flat_MES NORMALTOL=#mu_flat_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
*** COMMENT ********** // right
*** COMMENT ********** / rename body
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-018'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='outer_right' UNIQUENAME='outer_right'
*** COMMENT ********** / set material to air
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_right'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=#air_tot_DSL SIZE=#air_tot_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / face settings
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** / set face to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_right' IDENTIFIER=A.00010
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_flat_MES NORMALTOL=#mu_flat_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
*** COMMENT ********** // top
*** COMMENT ********** / rename body
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-025'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='outer_top' UNIQUENAME='outer_top'
*** COMMENT ********** / set material to air
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_top'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=#air_tot_DSL SIZE=#air_tot_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / face settings
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** / set face to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_top' IDENTIFIER=A.00010
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_floor' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_flat_MES NORMALTOL=#mu_flat_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
*** COMMENT ********** // bottom
*** COMMENT ********** / rename body
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-005'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='outer_bottom' UNIQUENAME='outer_bottom'
*** COMMENT ********** / set material to air
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_bottom'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=#air_tot_DSL SIZE=#air_tot_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / face settings
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** / set face to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_bottom' IDENTIFIER=A.00010
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_floor' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_flat_MES NORMALTOL=#mu_flat_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
*** COMMENT ********** /// brackets
*** COMMENT ********** // front-top
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-021'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='outer_brkt_front_top' UNIQUENAME='outer_brkt_front_top'
*** COMMENT ********** / set material to air
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_brkt_front_top'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=#air_tot_DSL SIZE=#air_tot_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / face settings
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** / set face to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_front_top' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_front_top' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_front_top' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
*** COMMENT ********** // front-bottom
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-003'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='outer_brkt_front_bottom' UNIQUENAME='outer_brkt_front_bottom'
*** COMMENT ********** / set material to air
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_brkt_front_bottom'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=#air_tot_DSL SIZE=#air_tot_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / face settings
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** / set face to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_front_bottom' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_front_bottom' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_front_bottom' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
*** COMMENT ********** // front-left
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-013'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='outer_brkt_front_left' UNIQUENAME='outer_brkt_front_left'
*** COMMENT ********** / set material to air
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_brkt_front_left'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=#air_tot_DSL SIZE=#air_tot_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / face settings
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** / set face to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_front_left' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_front_left' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_front_left' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
*** COMMENT ********** // front-right
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-017'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='outer_brkt_front_right' UNIQUENAME='outer_brkt_front_right'
*** COMMENT ********** / set material to air
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_brkt_front_right'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=#air_tot_DSL SIZE=#air_tot_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / face settings
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** / set face to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_front_right' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_front_right' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_front_right' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
*** COMMENT ********** // bottom-left
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-015'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='outer_brkt_bottom-left' UNIQUENAME='outer_brkt_bottom-left'
*** COMMENT ********** / set material to air
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_brkt_bottom-left'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=#air_tot_DSL SIZE=#air_tot_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / face settings
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** / set face to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_bottom-left' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_floor' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_bottom-left' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_floor' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_bottom-left' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_floor' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
*** COMMENT ********** // bottom-right
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-019'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='outer_brkt_bottom_right' UNIQUENAME='outer_brkt_bottom_right'
*** COMMENT ********** / set material to air
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_brkt_bottom_right'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=#air_tot_DSL SIZE=#air_tot_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / face settings
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** / set face to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_bottom_right' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_floor' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_bottom_right' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_floor' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_bottom_right' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_floor' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
*** COMMENT ********** // top-left
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-022'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='outer_brkt_top_left' UNIQUENAME='outer_brkt_top_left'
*** COMMENT ********** / set material to air
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_brkt_top_left'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=#air_tot_DSL SIZE=#air_tot_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / face settings
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** / set face to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_top_left' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_floor' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_top_left' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_floor' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_top_left' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_floor' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
*** COMMENT ********** // top-right
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-024'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='outer_brkt_top_right' UNIQUENAME='outer_brkt_top_right'
*** COMMENT ********** / set material to air
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_brkt_top_right'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=#air_tot_DSL SIZE=#air_tot_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / face settings
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** / set face to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_top_right' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_floor' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_top_right' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_floor' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_top_right' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_floor' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
*** COMMENT ********** // back-bottom
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-020'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='outer_brkt_back_bottom' UNIQUENAME='outer_brkt_back_bottom'
*** COMMENT ********** / set material to air
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_brkt_back_bottom'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=#air_tot_DSL SIZE=#air_tot_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / face settings
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** / set face to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_back_bottom' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_back_bottom' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_back_bottom' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
*** COMMENT ********** // back-top
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-023'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='outer_brkt_back_top' UNIQUENAME='outer_brkt_back_top'
*** COMMENT ********** / set material to air
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_brkt_back_top'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=#air_tot_DSL SIZE=#air_tot_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / face settings
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** / set face to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_back_top' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_back_top' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_back_top' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
*** COMMENT ********** // back-left
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-012'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='outer_brkt_back_left' UNIQUENAME='outer_brkt_back_left'
*** COMMENT ********** / set material to air
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_brkt_back_left'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=#air_tot_DSL SIZE=#air_tot_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / face settings
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** / set face to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_back_left' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_back_left' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_back_left' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
*** COMMENT ********** // back-right
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-016'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='outer_brkt_back_right' UNIQUENAME='outer_brkt_back_right'
*** COMMENT ********** / set material to air
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_brkt_back_right'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=#air_tot_DSL SIZE=#air_tot_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / face settings
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** / set face to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_back_right' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_back_right' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_back_right' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ******
*** COMMENT ********** // BH data
*** COMMENT ********** / create label
**** FILE INPUT ****** BHDATA OPTION=NEW  LABEL='outermumetal_door_BH'
**** FILE INPUT ****** BHDATA OPTION=NEW  LABEL='outermumetal_floor_BH'
**** FILE INPUT ******
*** COMMENT ********** / assign bh file to label
**** FILE INPUT ****** BHDATA OPTION=LOAD LABEL=outermumetal_door_BH FILE='bhdata//miniMSR_outer_door_anh.bh'
BHDATA label outermumetal_door_BH loaded with data from the file bhdata//miniMSR_outer_door_anh.bh
**** FILE INPUT ****** BHDATA OPTION=LOAD LABEL=outermumetal_floor_BH FILE='bhdata//miniMSR_outer_floor_anh.bh'
BHDATA label outermumetal_floor_BH loaded with data from the file bhdata//miniMSR_outer_floor_anh.bh
**** FILE INPUT ******
*** COMMENT ********** / setup boundary condition
**** FILE INPUT ****** BOUNDARY UNPICK
**** FILE INPUT ****** BOUNDARY GUIINIT
**** FILE INPUT ****** BOUNDARY PICK 'outer_mumetal_door'
**** FILE INPUT ****** BOUNDARY OPTION=MODIFY CONDITION=THINPLATE PLATEMATERIAL=outer_mumetal_door PLATETHICKNESS=0.002
**** FILE INPUT ****** BOUNDARY UNPICK
**** FILE INPUT ****** BOUNDARY UNPICK
**** FILE INPUT ****** BOUNDARY GUIINIT
**** FILE INPUT ****** BOUNDARY PICK 'outer_mumetal_floor'
**** FILE INPUT ****** BOUNDARY OPTION=MODIFY CONDITION=THINPLATE PLATEMATERIAL=outer_mumetal_floor PLATETHICKNESS=0.002
**** FILE INPUT ****** BOUNDARY UNPICK
**** FILE INPUT ******
*** COMMENT ********** / match material label to bh label
**** FILE INPUT ****** MATERIALS UNPICK
**** FILE INPUT ****** MATERIALS GUIINIT
**** FILE INPUT ****** MATERIALS PICK 'outer_mumetal_door'
**** FILE INPUT ****** MATERIALS OPTION=MODIFY MULINEARITY=NONLINEAR MUANISOTROPY=ISOTROPIC BH='outermumetal_door_BH'
**** FILE INPUT ****** MATERIALS UNPICK
**** FILE INPUT ****** MATERIALS UNPICK
**** FILE INPUT ****** MATERIALS GUIINIT
**** FILE INPUT ****** MATERIALS PICK 'outer_mumetal_floor'
**** FILE INPUT ****** MATERIALS OPTION=MODIFY MULINEARITY=NONLINEAR MUANISOTROPY=ISOTROPIC BH='outermumetal_floor_BH'
**** FILE INPUT ****** MATERIALS UNPICK
**** FILE INPUT ******
**** FILE INPUT ******
*** COMMENT ********** // create total air block
**** FILE INPUT ****** SKETCH OBJECT=BLOCK +COMPLETE
**** FILE INPUT ****** PREVIEW OPTION=END REDISPLAY=NO
**** FILE INPUT ****** BLOCK Name='Block' X0=0.5 Y0=0.5 Z0=0.5 X1=-0.5 Y1=-0.5 Z1=-0.5 MATERIALLABEL='Air' LEVEL=#air_tot_DSL
*** COMMENT ********** / rename
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Block'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='total_air_block' UNIQUENAME='total_air_block'
*** COMMENT ********** / mesh parameters
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='total_air_block'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=#air_tot_DSL SIZE=#air_tot_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
**** FILE INPUT ******
*** COMMENT ********** // create reduced air block
**** FILE INPUT ****** SKETCH OBJECT=BLOCK +COMPLETE
**** FILE INPUT ****** PREVIEW OPTION=END REDISPLAY=NO
**** FILE INPUT ****** BLOCK Name='Block' X0=2 Y0=2 Z0=2 X1=-2 Y1=-2 Z1=-2 MATERIALLABEL='Air' LEVEL=#air_red_DSL
*** COMMENT ********** / rename
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Block'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='reduced_air_block' UNIQUENAME='reduced_air_block'
*** COMMENT ********** / mesh parameters
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='reduced_air_block'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Reduced ELEMENTTYPE=Linear LEVEL=#air_red_DSL SIZE=#air_red_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
**** FILE INPUT ******
*** COMMENT ********** / zoom out view
**** FILE INPUT ****** THREED XORIGIN=0 YORIGIN=0 ZORIGIN=0 ROTX=20 ROTY=45 ROTZ=0 XASPECT=1 YASPECT=1 ZASPECT=1 SIZE=5 FACETANGLE=5 PERSPECTIVE=YES LINECOLOUR=YES OPTION=SETVIEW
**** FILE INPUT ******
*** COMMENT ********** //// Analysis Settings
**** FILE INPUT ****** MULTIPHYSICS OPTION=RESET
**** FILE INPUT ****** ANALYSISDATA OPTION=SET PROGRAM=TOSCAMAGN HX=0 HY=0 HZ=50 LINEAR=NO NITERATIONS=21 NLITERTYPE=NEWTON POTENTIALCUT=YES RHS=ADAPTIVE SCALEDRIVE=ALL TOLERANCE=0.001 USEDEFORMEDMESH=NO USEDIRECTSOLVER=NO
**** FILE INPUT ******
*** COMMENT ********** //// Symmetry
*** COMMENT ********** /++ BACKGROUND OPTION=LOAD
*** COMMENT ********** /++ BACKGROUND OPTION=SET SHAPE=BLOCK SCALEX=1 SCALEY=1 SCALEZ=1   XYSYMMETRYPLANE=NO YZSYMMETRYPLANE=YES ROTZNUM=1 ZXSYMMETRYPLANE=YES EMRYZ=TANGMAGN EMRZX=NORMMAGN
**** FILE INPUT ******
**** FILE INPUT ******
*** COMMENT ********** //// Create Model Body
**** FILE INPUT ****** MODEL CREATE
**** FILE INPUT ******
*** COMMENT ********** //// Meshing
*** COMMENT ********** / Surface Mesh
**** FILE INPUT ****** MESH GENERATOR=AUTOMATIC SIZE=0.3 NORMALTOL=30.0 SURFACETOL=0.0 TOLERANCE=1.0E-06 TYPE=PREFERTETRA
Mesh generator Type II
696 faces completed, generating 240554 surface elements

Triangular facet quality on 696 faces
Triangles: 240554, Max: 1.0, Min: 0.159128, Mean: 0.928732, Power2Mean: 0.818214

**** FILE INPUT ******
*** COMMENT ********** / Volume Mesh
**** FILE INPUT ****** FILL TOL=1.0E-06

Tetrahedral element quality on 38 cells
Tetrahedra: 1132253, Max: 1.631634, Min: 0.0850922, RMS: 1.017677, Power2Mean: 0.874039, Worst cell: 34

Total nodes:    192548
Tetrahedra:     1132253
Total volume: 64.0
**** FILE INPUT ******
*** COMMENT ********** //// Create Database
**** FILE INPUT ****** SOLVERS SOLVENOW=NO SAVEMODEL=YES,
**** FILE INPUT ****** SOLVERS OPTION=TEST FILE='mini_MSR' UNITS=METRE ELEMENT=MIXED SURFACE=CURVED
**** FILE INPUT ****** COMMENT CLEAR=YES TYPE=DBTITLE
**** FILE INPUT ****** SOLVERS OPTION=OVERWRITE
Opening database mini_MSR.op3, simulation number 0 on 23/May/2024 at 12:20:15
   Magnetostatic Analysis (TOSCA)

   File: mini_MSR.op3 simulation:  1
   Created on: 23/May/2024 12:20:00
   In Directory:
   C:\Users\mzhao\OPERA_miniMSR\
   By Machine:
   Node: UCNENG01. Processor: Intel64 Family 6 Model 42 Stepping 7, GenuineIntel. System: Windows 10 Version 2009. version windows (10.0.19045)
   Log Files: Modeller_202405230944315521.log/lp
   Simulation created using: Opera-3d/Modeller Version 21.3.3
   User did not enter a title

   SI units

   No conductors
   External field vector (0.0,0.0,50.0 A/m)
   Adaptive RHS integrals

Warning: Automatic potential cuts have been disabled because there are thin plates.

   Boundary conditions:
     TANGMAGN
        Note: Assumed as the default boundary condition to one or more external faces.
     THINPLATE
        thin plate of material OUTER_MUMETAL_FLOOR, thickness: 2.0E-03 [m]
        thin plate of material OUTER_MUMETAL_DOOR, thickness: 2.0E-03 [m]
        thin plate of material INNER_MUMETAL_FLOOR, thickness: 1.5E-03 [m]
        thin plate of material INNER_MUMETAL_DOOR, thickness: 1.5E-03 [m]

   Nonlinear iteration data
   Newton-Raphson Iterations
   Maximum Iterations: 21
   Tolerance: 1.0E-03

   No periodicity conditions have been defined

   1 solution scaling factor defined for all drives: 1.0

   Materials defined:
   OUTER_MUMETAL_FLOOR
     Isotropic permeability: bhdata//miniMSR_outer_floor_anh.bh (15193.9 )
   OUTER_MUMETAL_DOOR
     Isotropic permeability: bhdata//miniMSR_outer_door_anh.bh (5933.308)
   INNER_MUMETAL_FLOOR
     Isotropic permeability: bhdata//miniMSR_inner_floor_anh.bh (25009.79)
   INNER_MUMETAL_DOOR
     Isotropic permeability: 1.0
   AIR

   807914 nodes in the model
   1686965 edges in the model
   Both linear and quadratic elements exist in the model
   1132253 tetrahedral elements (872092 linear and 260161 quadratic)
   1132253 elements in total

   Symmetry:
   none

WARNINGS:
1 warning when preparing simulation
**** FILE INPUT ******
**** FILE INPUT ******
**** FILE INPUT ******
**** FILE INPUT ******
**** FILE INPUT ******
End of command file.
*** CONTROL COMMAND ** $ END COMI
**** COMMAND INPUT ***
*** CONTROL COMMAND ** $ COMI 'C:\Users\mzhao\OPERA_miniMSR\test.comi'
Opening file for input: C:\Users\mzhao\OPERA_miniMSR\test.comi
*** COMMENT ********** // this script creates model of mini-MSR in OPERA
*** COMMENT ********** // author: M. Zhao
*** COMMENT ********** // May 2024
*** CONTROL COMMAND ** $string yesorno YES
 yesorno='YES'
**** FILE INPUT ****** CLEAR REVERT=NO
Data changes within the model will be lost.
(this operation cannot be UNDONE)
Continue?
The default value has been used from the YESORNO string variable (YES)
Command language initialization.
**** FILE INPUT ****** THREED XORIGIN=0 YORIGIN=0 ZORIGIN=0 ROTX=0 ROTY=0 ROTZ=0 XASPECT=1 YASPECT=1 ZASPECT=1 SIZE=0 FACETANGLE=5 PERSPECTIVE=YES LINECOLOUR=YES OPTION=SETVIEW
**** FILE INPUT ******
*** COMMENT ********** ////////////////////////////////////////////////////////////
*** COMMENT ********** / Constants
*** COMMENT ********** ////////////////////////////////////////////////////////////
*** COMMENT ********** //Data Storage Levels
*** COMMENT ********** / mumetal
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#mu_DSL, Value=500, DESCRIPTION='data storage level for mumetal'
Variable      Value            Expression
#mu_DSL       500.0            500
Description: data storage level for mumetal
*** COMMENT ********** / air (total potential)
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#air_tot_DSL, Value=200, DESCRIPTION='data storage level for total potential air'
Variable      Value            Expression
#air_tot_DSL  200.0            200
Description: data storage level for total potential air
*** COMMENT ********** / air (reduced potential)
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#air_red_DSL, Value=100, DESCRIPTION='data storage level for reduced potential air'
Variable      Value            Expression
#air_red_DSL  100.0            100
Description: data storage level for reduced potential air
**** FILE INPUT ******
*** COMMENT ********** /// meshing parameters
*** COMMENT ********** // max element size (MES)
*** COMMENT ********** / flat mumetal
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#mu_flat_MES, Value=0.05, DESCRIPTION='maximum element size for flat mumetal'
Variable      Value            Expression
#mu_flat_MES  0.05             0.05
Description: maximum element size for flat mumetal
*** COMMENT ********** / bent mumetal
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#mu_bent_MES, Value=0.03, DESCRIPTION='maximum element size for bent mumetal'
Variable      Value            Expression
#mu_bent_MES  0.03             0.03
Description: maximum element size for bent mumetal
*** COMMENT ********** / air (total potential)
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#air_tot_MES, Value=0.02, DESCRIPTION='maximum element size for total potential air'
Variable      Value            Expression
#air_tot_MES  0.02             0.02
Description: maximum element size for total potential air
*** COMMENT ********** / air (reduced potential)
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#air_red_MES, Value=0.3, DESCRIPTION='maximum element size for reduced potential air'
Variable      Value            Expression
#air_red_MES  0.3              0.3
Description: maximum element size for reduced potential air
**** FILE INPUT ******
*** COMMENT ********** // maximum angle between elements (MAbE)
*** COMMENT ********** / flat mumetal
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#mu_flat_MAbE, Value=30, DESCRIPTION='maximum angle between elements for flat mumetal'
Variable      Value            Expression
#mu_flat_MAbE 30.0             30
Description: maximum angle between elements for flat mumetal
*** COMMENT ********** / bent mumetal
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#mu_bent_MAbE, Value=30, DESCRIPTION='maximum angle between elements for bent mumetal'
Variable      Value            Expression
#mu_bent_MAbE 30.0             30
Description: maximum angle between elements for bent mumetal
*** COMMENT ********** / air
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#air_MAbE, Value=30, DESCRIPTION='maximum angle between elements for air'
Variable      Value            Expression
#air_MAbE     30.0             30
Description: maximum angle between elements for air
**** FILE INPUT ******
**** FILE INPUT ******
*** COMMENT ********** ////////////////////////////////////////////////////////////
*** COMMENT ********** / Modelling
*** COMMENT ********** ////////////////////////////////////////////////////////////
**** FILE INPUT ******
*** COMMENT ********** //// Inner Layer
**** FILE INPUT ******
*** COMMENT ********** / load structural step file
**** FILE INPUT ****** LOAD OPTION=INSERT FILE='stpfiles//inner_layer.STEP'
Imported material name from label: layer1_NAUO1#inner_flat_front_Boss_Extrude2
Imported material name from label: layer1_NAUO2#inner_bracket_Cut_Extrude3
Imported material name from label: layer1_NAUO3#inner_bracket_Cut_Extrude3
Imported material name from label: layer1_NAUO4#inner_bracket_Cut_Extrude3
Imported material name from label: layer1_NAUO5#inner_bracket_Cut_Extrude3
Imported material name from label: layer1_NAUO6#inner_flat_Boss_Extrude2
Imported material name from label: layer1_NAUO7#inner_flat_Boss_Extrude2
Imported material name from label: layer1_NAUO8#inner_bracket_Cut_Extrude3
Imported material name from label: layer1_NAUO9#inner_bracket_Cut_Extrude3
Imported material name from label: layer1_NAUO10#inner_flat_Boss_Extrude2
Imported material name from label: layer1_NAUO11#inner_bracket_Cut_Extrude3
Imported material name from label: layer1_NAUO12#inner_bracket_Cut_Extrude3
Imported material name from label: layer1_NAUO13#inner_bracket_Cut_Extrude3
Imported material name from label: layer1_NAUO14#inner_bracket_Cut_Extrude3
Imported material name from label: layer1_NAUO15#inner_bracket_Cut_Extrude3
Imported material name from label: layer1_NAUO16#inner_bracket_Cut_Extrude3
Imported material name from label: layer1_NAUO17#inner_flat_Boss_Extrude2
Imported material name from label: layer1_NAUO18#inner_flat_Boss_Extrude2
**** FILE INPUT ******
*** COMMENT ********** /// Flat panels
*** COMMENT ********** // front
*** COMMENT ********** / rename body
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-001'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_front' UNIQUENAME='inner_front'
*** COMMENT ********** / set material to air
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_front'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=#air_tot_DSL SIZE=#air_tot_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / face settings
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** / set face to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_front' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_flat_MES NORMALTOL=#mu_flat_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
*** COMMENT ********** // back
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-018'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_back' UNIQUENAME='inner_back'
*** COMMENT ********** / set material to air
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_back'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=#air_tot_DSL SIZE=#air_tot_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / face settings
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** / set face to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_back' IDENTIFIER=A.00010
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_flat_MES NORMALTOL=#mu_flat_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
*** COMMENT ********** // left
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-010'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_left' UNIQUENAME='inner_left'
*** COMMENT ********** / set material to air
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_left'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=#air_tot_DSL SIZE=#air_tot_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / face settings
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** / set face to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_left' IDENTIFIER=A.00010
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_flat_MES NORMALTOL=#mu_flat_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
*** COMMENT ********** // right
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-006'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_right' UNIQUENAME='inner_right'
*** COMMENT ********** / set material to air
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_right'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=#air_tot_DSL SIZE=#air_tot_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / face settings
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** / set face to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_right' IDENTIFIER=A.00010
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_flat_MES NORMALTOL=#mu_flat_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
*** COMMENT ********** // top
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-017'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_top' UNIQUENAME='inner_top'
*** COMMENT ********** / set material to air
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_top'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=#air_tot_DSL SIZE=#air_tot_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / face settings
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** / set face to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_top' IDENTIFIER=A.00010
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_floor' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_flat_MES NORMALTOL=#mu_flat_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
*** COMMENT ********** // bottom
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-007'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_bottom' UNIQUENAME='inner_bottom'
*** COMMENT ********** / set material to air
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_bottom'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=#air_tot_DSL SIZE=#air_tot_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / face settings
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** / set face to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_bottom' IDENTIFIER=A.00010
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_floor' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_flat_MES NORMALTOL=#mu_flat_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ******
*** COMMENT ********** /// brackets
*** COMMENT ********** // front-top
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-005'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_brkt_front_top' UNIQUENAME='inner_brkt_front_top'
*** COMMENT ********** / set material to air
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_brkt_front_top'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=#air_tot_DSL SIZE=#air_tot_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / face settings
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** / set face to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_front_top' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_front_top' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_front_top' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
*** COMMENT ********** // front-bottom
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-003'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_brkt_front_bottom' UNIQUENAME='inner_brkt_front_bottom'
*** COMMENT ********** / set material to air
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_brkt_front_bottom'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=#air_tot_DSL SIZE=#air_tot_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / face settings
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** / set face to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_front_bottom' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_front_bottom' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_front_bottom' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
*** COMMENT ********** // front-left
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-004'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_brkt_front_left' UNIQUENAME='inner_brkt_front_left'
*** COMMENT ********** / set material to air
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_brkt_front_left'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=#air_tot_DSL SIZE=#air_tot_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / face settings
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** / set face to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_front_left' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_front_left' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_front_left' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
*** COMMENT ********** // front-right
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-002'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_brkt_front_right' UNIQUENAME='inner_brkt_front_right'
*** COMMENT ********** / set material to air
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_brkt_front_right'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=#air_tot_DSL SIZE=#air_tot_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / face settings
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** / set face to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_front_right' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_front_right' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_front_right' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
*** COMMENT ********** // bottom-left
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-012'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_brkt_bottom-left' UNIQUENAME='inner_brkt_bottom-left'
*** COMMENT ********** / set material to air
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_brkt_bottom-left'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=#air_tot_DSL SIZE=#air_tot_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / face settings
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** / set face to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_bottom-left' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_floor' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_bottom-left' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_floor' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_bottom-left' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_floor' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
*** COMMENT ********** // bottom-right
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-008'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_brkt_bottom_right' UNIQUENAME='inner_brkt_bottom_right'
*** COMMENT ********** / set material to air
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_brkt_bottom_right'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=#air_tot_DSL SIZE=#air_tot_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / face settings
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** / set face to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_bottom_right' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_floor' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_bottom_right' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_floor' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_bottom_right' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_floor' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
*** COMMENT ********** // top-left
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-011'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_brkt_top_left' UNIQUENAME='inner_brkt_top_left'
*** COMMENT ********** / set material to air
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_brkt_top_left'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=#air_tot_DSL SIZE=#air_tot_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / face settings
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** / set face to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_top_left' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_floor' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_top_left' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_floor' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_top_left' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_floor' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
*** COMMENT ********** // top-right
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-009'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_brkt_top_right' UNIQUENAME='inner_brkt_top_right'
*** COMMENT ********** / set material to air
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_brkt_top_right'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=#air_tot_DSL SIZE=#air_tot_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / face settings
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** / set face to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_top_right' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_floor' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_top_right' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_floor' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_top_right' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_floor' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
*** COMMENT ********** // back-bottom
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-013'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_brkt_back_bottom' UNIQUENAME='inner_brkt_back_bottom'
*** COMMENT ********** / set material to air
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_brkt_back_bottom'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=#air_tot_DSL SIZE=#air_tot_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / face settings
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** / set face to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_back_bottom' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_back_bottom' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_back_bottom' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
*** COMMENT ********** // back-top
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-015'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_brkt_back_top' UNIQUENAME='inner_brkt_back_top'
*** COMMENT ********** / set material to air
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_brkt_back_top'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=#air_tot_DSL SIZE=#air_tot_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / face settings
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** / set face to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_back_top' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_back_top' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_back_top' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
*** COMMENT ********** // back-left
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-016'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_brkt_back_left' UNIQUENAME='inner_brkt_back_left'
*** COMMENT ********** / set material to air
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_brkt_back_left'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=#air_tot_DSL SIZE=#air_tot_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / face settings
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** / set face to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_back_left' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_back_left' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_back_left' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
*** COMMENT ********** // back-right
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-014'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_brkt_back_right' UNIQUENAME='inner_brkt_back_right'
*** COMMENT ********** / set material to air
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_brkt_back_right'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=#air_tot_DSL SIZE=#air_tot_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / face settings
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** / set face to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_back_right' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_back_right' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_back_right' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ******
**** FILE INPUT ******
*** COMMENT ********** // BH data
*** COMMENT ********** / create labels
**** FILE INPUT ****** BHDATA OPTION=NEW  LABEL='innermumetal_door_BH'
**** FILE INPUT ****** BHDATA OPTION=NEW  LABEL='innermumetal_floor_BH'
**** FILE INPUT ******
*** COMMENT ********** / assign bh file to label
**** FILE INPUT ****** BHDATA OPTION=LOAD LABEL=innermumetal_door_BH FILE='bhdata//miniMSR_inner_door_anh.bh'
BHDATA label innermumetal_door_BH loaded with data from the file bhdata//miniMSR_inner_door_anh.bh
**** FILE INPUT ****** BHDATA OPTION=LOAD LABEL=innermumetal_floor_BH FILE='bhdata//miniMSR_inner_floor_anh.bh'
BHDATA label innermumetal_floor_BH loaded with data from the file bhdata//miniMSR_inner_floor_anh.bh
**** FILE INPUT ******
*** COMMENT ********** / setup boundary condition
**** FILE INPUT ****** BOUNDARY UNPICK
**** FILE INPUT ****** BOUNDARY GUIINIT
**** FILE INPUT ****** BOUNDARY PICK 'inner_mumetal_door'
**** FILE INPUT ****** BOUNDARY OPTION=MODIFY CONDITION=THINPLATE PLATEMATERIAL=inner_mumetal_door PLATETHICKNESS=0.0015
**** FILE INPUT ****** BOUNDARY UNPICK
**** FILE INPUT ******
**** FILE INPUT ****** BOUNDARY UNPICK
**** FILE INPUT ****** BOUNDARY GUIINIT
**** FILE INPUT ****** BOUNDARY PICK 'inner_mumetal_floor'
**** FILE INPUT ****** BOUNDARY OPTION=MODIFY CONDITION=THINPLATE PLATEMATERIAL=inner_mumetal_floor PLATETHICKNESS=0.0015
**** FILE INPUT ****** BOUNDARY UNPICK
**** FILE INPUT ******
*** COMMENT ********** / match material label to bh label
**** FILE INPUT ****** MATERIALS UNPICK
**** FILE INPUT ****** MATERIALS GUIINIT
**** FILE INPUT ****** MATERIALS PICK 'inner_mumetal_door'
**** FILE INPUT ****** MATERIALS OPTION=MODIFY MULINEARITY=NONLINEAR MUANISOTROPY=ISOTROPIC BH='innermumetal_door_BH'
**** FILE INPUT ****** MATERIALS UNPICK
**** FILE INPUT ****** MATERIALS UNPICK
**** FILE INPUT ****** MATERIALS GUIINIT
**** FILE INPUT ****** MATERIALS PICK 'inner_mumetal_floor'
**** FILE INPUT ****** MATERIALS OPTION=MODIFY MULINEARITY=NONLINEAR MUANISOTROPY=ISOTROPIC BH='innermumetal_floor_BH'
**** FILE INPUT ****** MATERIALS UNPICK
**** FILE INPUT ******
**** FILE INPUT ******
*** COMMENT ********** //// Outer Layer
**** FILE INPUT ******
*** COMMENT ********** / load structural step file
**** FILE INPUT ****** LOAD OPTION=INSERT FILE='stpfiles//outer_layer.STEP'
Imported material name from label: outer_layer_NAUO1#outer_flat_front_Boss_Extrude2
Imported material name from label: outer_layer_NAUO2#outer_bracket_Cut_Extrude4
Imported material name from label: outer_layer_NAUO3#outer_bracket_Cut_Extrude4
Imported material name from label: outer_layer_NAUO4#outer_flat_Boss_Extrude2
Imported material name from label: outer_layer_NAUO5#outer_flat_Boss_Extrude2
Imported material name from label: outer_layer_NAUO6#outer_bracket_Cut_Extrude4
Imported material name from label: outer_layer_NAUO7#outer_bracket_Cut_Extrude4
Imported material name from label: outer_layer_NAUO8#outer_bracket_Cut_Extrude4
Imported material name from label: outer_layer_NAUO9#outer_bracket_Cut_Extrude4
Imported material name from label: outer_layer_NAUO10#outer_flat_Boss_Extrude2
Imported material name from label: outer_layer_NAUO11#outer_bracket_Cut_Extrude4
Imported material name from label: outer_layer_NAUO12#outer_bracket_Cut_Extrude4
Imported material name from label: outer_layer_NAUO13#outer_bracket_Cut_Extrude4
Imported material name from label: outer_layer_NAUO14#outer_flat_Boss_Extrude2
Imported material name from label: outer_layer_NAUO15#outer_bracket_Cut_Extrude4
Imported material name from label: outer_layer_NAUO16#outer_bracket_Cut_Extrude4
Imported material name from label: outer_layer_NAUO17#outer_bracket_Cut_Extrude4
Imported material name from label: outer_layer_NAUO18#outer_flat_Boss_Extrude2
Imported material name from label: outer_layer_NAUO19#outer_bracket_Cut_Extrude4
Imported material name from label: outer_layer_NAUO20#outer_bracket_Cut_Extrude4
Imported material name from label: outer_layer_NAUO21#outer_bracket_Cut_Extrude4
Imported material name from label: outer_layer_NAUO22#outer_bracket_Cut_Extrude4
Imported material name from label: outer_layer_NAUO23#outer_bracket_Cut_Extrude4
Imported material name from label: outer_layer_NAUO24#outer_bracket_Cut_Extrude4
Imported material name from label: outer_layer_NAUO25#outer_flat_Boss_Extrude2
Imported material name from label: outer_layer_NAUO26#outer_flat_Boss_Extrude2
**** FILE INPUT ******
*** COMMENT ********** /// Flat panels
*** COMMENT ********** // front
*** COMMENT ********** / rename body
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-001'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='outer_front' UNIQUENAME='outer_front'
*** COMMENT ********** / set material to air
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_front'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=#air_tot_DSL SIZE=#air_tot_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / face settings
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** / set face to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_front' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_flat_MES NORMALTOL=#mu_flat_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
*** COMMENT ********** // back
*** COMMENT ********** / rename body
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-026'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='outer_back' UNIQUENAME='outer_back'
*** COMMENT ********** / set material to air
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_back'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=#air_tot_DSL SIZE=#air_tot_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / face settings
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** / set face to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_back' IDENTIFIER=A.00010
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_flat_MES NORMALTOL=#mu_flat_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
*** COMMENT ********** // left
*** COMMENT ********** / rename body
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-014'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='outer_left' UNIQUENAME='outer_left'
*** COMMENT ********** / set material to air
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_left'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=#air_tot_DSL SIZE=#air_tot_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / face settings
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** / set face to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_left' IDENTIFIER=A.00010
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_flat_MES NORMALTOL=#mu_flat_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
*** COMMENT ********** // right
*** COMMENT ********** / rename body
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-018'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='outer_right' UNIQUENAME='outer_right'
*** COMMENT ********** / set material to air
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_right'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=#air_tot_DSL SIZE=#air_tot_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / face settings
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** / set face to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_right' IDENTIFIER=A.00010
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_flat_MES NORMALTOL=#mu_flat_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
*** COMMENT ********** // top
*** COMMENT ********** / rename body
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-025'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='outer_top' UNIQUENAME='outer_top'
*** COMMENT ********** / set material to air
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_top'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=#air_tot_DSL SIZE=#air_tot_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / face settings
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** / set face to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_top' IDENTIFIER=A.00010
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_floor' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_flat_MES NORMALTOL=#mu_flat_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
*** COMMENT ********** // bottom
*** COMMENT ********** / rename body
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-005'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='outer_bottom' UNIQUENAME='outer_bottom'
*** COMMENT ********** / set material to air
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_bottom'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=#air_tot_DSL SIZE=#air_tot_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / face settings
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** / set face to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_bottom' IDENTIFIER=A.00010
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_floor' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_flat_MES NORMALTOL=#mu_flat_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
*** COMMENT ********** /// brackets
*** COMMENT ********** // front-top
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-021'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='outer_brkt_front_top' UNIQUENAME='outer_brkt_front_top'
*** COMMENT ********** / set material to air
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_brkt_front_top'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=#air_tot_DSL SIZE=#air_tot_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / face settings
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** / set face to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_front_top' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_front_top' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_front_top' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
*** COMMENT ********** // front-bottom
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-003'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='outer_brkt_front_bottom' UNIQUENAME='outer_brkt_front_bottom'
*** COMMENT ********** / set material to air
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_brkt_front_bottom'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=#air_tot_DSL SIZE=#air_tot_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / face settings
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** / set face to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_front_bottom' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_front_bottom' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_front_bottom' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
*** COMMENT ********** // front-left
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-013'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='outer_brkt_front_left' UNIQUENAME='outer_brkt_front_left'
*** COMMENT ********** / set material to air
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_brkt_front_left'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=#air_tot_DSL SIZE=#air_tot_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / face settings
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** / set face to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_front_left' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_front_left' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_front_left' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
*** COMMENT ********** // front-right
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-017'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='outer_brkt_front_right' UNIQUENAME='outer_brkt_front_right'
*** COMMENT ********** / set material to air
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_brkt_front_right'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=#air_tot_DSL SIZE=#air_tot_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / face settings
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** / set face to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_front_right' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_front_right' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_front_right' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
*** COMMENT ********** // bottom-left
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-015'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='outer_brkt_bottom-left' UNIQUENAME='outer_brkt_bottom-left'
*** COMMENT ********** / set material to air
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_brkt_bottom-left'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=#air_tot_DSL SIZE=#air_tot_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / face settings
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** / set face to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_bottom-left' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_floor' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_bottom-left' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_floor' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_bottom-left' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_floor' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
*** COMMENT ********** // bottom-right
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-019'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='outer_brkt_bottom_right' UNIQUENAME='outer_brkt_bottom_right'
*** COMMENT ********** / set material to air
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_brkt_bottom_right'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=#air_tot_DSL SIZE=#air_tot_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / face settings
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** / set face to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_bottom_right' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_floor' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_bottom_right' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_floor' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_bottom_right' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_floor' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
*** COMMENT ********** // top-left
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-022'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='outer_brkt_top_left' UNIQUENAME='outer_brkt_top_left'
*** COMMENT ********** / set material to air
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_brkt_top_left'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=#air_tot_DSL SIZE=#air_tot_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / face settings
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** / set face to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_top_left' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_floor' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_top_left' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_floor' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_top_left' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_floor' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
*** COMMENT ********** // top-right
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-024'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='outer_brkt_top_right' UNIQUENAME='outer_brkt_top_right'
*** COMMENT ********** / set material to air
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_brkt_top_right'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=#air_tot_DSL SIZE=#air_tot_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / face settings
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** / set face to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_top_right' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_floor' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_top_right' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_floor' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_top_right' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_floor' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
*** COMMENT ********** // back-bottom
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-020'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='outer_brkt_back_bottom' UNIQUENAME='outer_brkt_back_bottom'
*** COMMENT ********** / set material to air
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_brkt_back_bottom'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=#air_tot_DSL SIZE=#air_tot_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / face settings
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** / set face to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_back_bottom' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_back_bottom' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_back_bottom' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
*** COMMENT ********** // back-top
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-023'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='outer_brkt_back_top' UNIQUENAME='outer_brkt_back_top'
*** COMMENT ********** / set material to air
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_brkt_back_top'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=#air_tot_DSL SIZE=#air_tot_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / face settings
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** / set face to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_back_top' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_back_top' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_back_top' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
*** COMMENT ********** // back-left
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-012'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='outer_brkt_back_left' UNIQUENAME='outer_brkt_back_left'
*** COMMENT ********** / set material to air
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_brkt_back_left'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=#air_tot_DSL SIZE=#air_tot_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / face settings
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** / set face to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_back_left' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_back_left' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_back_left' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
*** COMMENT ********** // back-right
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-016'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='outer_brkt_back_right' UNIQUENAME='outer_brkt_back_right'
*** COMMENT ********** / set material to air
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_brkt_back_right'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=#air_tot_DSL SIZE=#air_tot_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / face settings
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** / set face to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_back_right' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_back_right' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_back_right' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ******
*** COMMENT ********** // BH data
*** COMMENT ********** / create label
**** FILE INPUT ****** BHDATA OPTION=NEW  LABEL='outermumetal_door_BH'
**** FILE INPUT ****** BHDATA OPTION=NEW  LABEL='outermumetal_floor_BH'
**** FILE INPUT ******
*** COMMENT ********** / assign bh file to label
**** FILE INPUT ****** BHDATA OPTION=LOAD LABEL=outermumetal_door_BH FILE='bhdata//miniMSR_outer_door_anh.bh'
BHDATA label outermumetal_door_BH loaded with data from the file bhdata//miniMSR_outer_door_anh.bh
**** FILE INPUT ****** BHDATA OPTION=LOAD LABEL=outermumetal_floor_BH FILE='bhdata//miniMSR_outer_floor_anh.bh'
BHDATA label outermumetal_floor_BH loaded with data from the file bhdata//miniMSR_outer_floor_anh.bh
**** FILE INPUT ******
*** COMMENT ********** / setup boundary condition
**** FILE INPUT ****** BOUNDARY UNPICK
**** FILE INPUT ****** BOUNDARY GUIINIT
**** FILE INPUT ****** BOUNDARY PICK 'outer_mumetal_door'
**** FILE INPUT ****** BOUNDARY OPTION=MODIFY CONDITION=THINPLATE PLATEMATERIAL=outer_mumetal_door PLATETHICKNESS=0.002
**** FILE INPUT ****** BOUNDARY UNPICK
**** FILE INPUT ****** BOUNDARY UNPICK
**** FILE INPUT ****** BOUNDARY GUIINIT
**** FILE INPUT ****** BOUNDARY PICK 'outer_mumetal_floor'
**** FILE INPUT ****** BOUNDARY OPTION=MODIFY CONDITION=THINPLATE PLATEMATERIAL=outer_mumetal_floor PLATETHICKNESS=0.002
**** FILE INPUT ****** BOUNDARY UNPICK
**** FILE INPUT ******
*** COMMENT ********** / match material label to bh label
**** FILE INPUT ****** MATERIALS UNPICK
**** FILE INPUT ****** MATERIALS GUIINIT
**** FILE INPUT ****** MATERIALS PICK 'outer_mumetal_door'
**** FILE INPUT ****** MATERIALS OPTION=MODIFY MULINEARITY=NONLINEAR MUANISOTROPY=ISOTROPIC BH='outermumetal_door_BH'
**** FILE INPUT ****** MATERIALS UNPICK
**** FILE INPUT ****** MATERIALS UNPICK
**** FILE INPUT ****** MATERIALS GUIINIT
**** FILE INPUT ****** MATERIALS PICK 'outer_mumetal_floor'
**** FILE INPUT ****** MATERIALS OPTION=MODIFY MULINEARITY=NONLINEAR MUANISOTROPY=ISOTROPIC BH='outermumetal_floor_BH'
**** FILE INPUT ****** MATERIALS UNPICK
**** FILE INPUT ******
**** FILE INPUT ******
*** COMMENT ********** // create total air block
**** FILE INPUT ****** SKETCH OBJECT=BLOCK +COMPLETE
**** FILE INPUT ****** PREVIEW OPTION=END REDISPLAY=NO
**** FILE INPUT ****** BLOCK Name='Block' X0=0.5 Y0=0.5 Z0=0.5 X1=-0.5 Y1=-0.5 Z1=-0.5 MATERIALLABEL='Air' LEVEL=#air_tot_DSL
*** COMMENT ********** / rename
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Block'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='total_air_block' UNIQUENAME='total_air_block'
*** COMMENT ********** / mesh parameters
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='total_air_block'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=#air_tot_DSL SIZE=#air_tot_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
**** FILE INPUT ******
*** COMMENT ********** // create reduced air block
**** FILE INPUT ****** SKETCH OBJECT=BLOCK +COMPLETE
**** FILE INPUT ****** PREVIEW OPTION=END REDISPLAY=NO
**** FILE INPUT ****** BLOCK Name='Block' X0=2 Y0=2 Z0=2 X1=-2 Y1=-2 Z1=-2 MATERIALLABEL='Air' LEVEL=#air_red_DSL
*** COMMENT ********** / rename
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Block'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='reduced_air_block' UNIQUENAME='reduced_air_block'
*** COMMENT ********** / mesh parameters
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='reduced_air_block'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Reduced ELEMENTTYPE=Linear LEVEL=#air_red_DSL SIZE=#air_red_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
**** FILE INPUT ******
*** COMMENT ********** / zoom out view
**** FILE INPUT ****** THREED XORIGIN=0 YORIGIN=0 ZORIGIN=0 ROTX=20 ROTY=45 ROTZ=0 XASPECT=1 YASPECT=1 ZASPECT=1 SIZE=5 FACETANGLE=5 PERSPECTIVE=YES LINECOLOUR=YES OPTION=SETVIEW
**** FILE INPUT ******
*** COMMENT ********** //// Analysis Settings
**** FILE INPUT ****** MULTIPHYSICS OPTION=RESET
**** FILE INPUT ****** ANALYSISDATA OPTION=SET PROGRAM=TOSCAMAGN HX=0 HY=0 HZ=50 LINEAR=NO NITERATIONS=21 NLITERTYPE=NEWTON POTENTIALCUT=YES RHS=ADAPTIVE SCALEDRIVE=ALL TOLERANCE=0.001 USEDEFORMEDMESH=NO USEDIRECTSOLVER=NO
**** FILE INPUT ******
*** COMMENT ********** //// Symmetry
*** COMMENT ********** /++ BACKGROUND OPTION=LOAD
*** COMMENT ********** /++ BACKGROUND OPTION=SET SHAPE=BLOCK SCALEX=1 SCALEY=1 SCALEZ=1   XYSYMMETRYPLANE=NO YZSYMMETRYPLANE=YES ROTZNUM=1 ZXSYMMETRYPLANE=YES EMRYZ=TANGMAGN EMRZX=NORMMAGN
**** FILE INPUT ******
**** FILE INPUT ******
*** COMMENT ********** //// Create Model Body
**** FILE INPUT ****** MODEL CREATE
**** FILE INPUT ******
*** COMMENT ********** //// Meshing
*** COMMENT ********** / Surface Mesh
**** FILE INPUT ****** MESH GENERATOR=AUTOMATIC SIZE=0.3 NORMALTOL=30.0 SURFACETOL=0.0 TOLERANCE=1.0E-06 TYPE=PREFERTETRA
Mesh generator Type II
696 faces completed, generating 240380 surface elements

Triangular facet quality on 696 faces
Triangles: 240380, Max: 1.0, Min: 0.159128, Mean: 0.928815, Power2Mean: 0.818109

**** FILE INPUT ******
*** COMMENT ********** / Volume Mesh
**** FILE INPUT ****** FILL TOL=1.0E-06
Mesh Abort :: user abort.
Command file processing aborted
Aborted file: 'C:\Users\mzhao\OPERA_miniMSR\test.comi' (Line 779)
Mesh Generator Type II Error Number : 5
 Error Message : Failed to generate tet mesh: user interrupt
**** COMMAND INPUT ***
*** CONTROL COMMAND ** $ COMI 'C:\Users\mzhao\OPERA_miniMSR\test.comi'
Opening file for input: C:\Users\mzhao\OPERA_miniMSR\test.comi
*** COMMENT ********** // this script creates model of mini-MSR in OPERA
*** COMMENT ********** // author: M. Zhao
*** COMMENT ********** // May 2024
*** CONTROL COMMAND ** $string yesorno YES
 yesorno='YES'
**** FILE INPUT ****** CLEAR REVERT=NO
Data changes within the model will be lost.
(this operation cannot be UNDONE)
Continue?
The default value has been used from the YESORNO string variable (YES)
Command language initialization.
**** FILE INPUT ****** THREED XORIGIN=0 YORIGIN=0 ZORIGIN=0 ROTX=0 ROTY=0 ROTZ=0 XASPECT=1 YASPECT=1 ZASPECT=1 SIZE=0 FACETANGLE=5 PERSPECTIVE=YES LINECOLOUR=YES OPTION=SETVIEW
**** FILE INPUT ******
*** COMMENT ********** ////////////////////////////////////////////////////////////
*** COMMENT ********** / Constants
*** COMMENT ********** ////////////////////////////////////////////////////////////
*** COMMENT ********** //Data Storage Levels
*** COMMENT ********** / mumetal
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#mu_DSL, Value=500, DESCRIPTION='data storage level for mumetal'
Variable      Value            Expression
#mu_DSL       500.0            500
Description: data storage level for mumetal
*** COMMENT ********** / air (total potential)
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#air_tot_DSL, Value=200, DESCRIPTION='data storage level for total potential air'
Variable      Value            Expression
#air_tot_DSL  200.0            200
Description: data storage level for total potential air
*** COMMENT ********** / air (reduced potential)
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#air_red_DSL, Value=100, DESCRIPTION='data storage level for reduced potential air'
Variable      Value            Expression
#air_red_DSL  100.0            100
Description: data storage level for reduced potential air
**** FILE INPUT ******
*** COMMENT ********** /// meshing parameters
*** COMMENT ********** // max element size (MES)
*** COMMENT ********** / flat mumetal
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#mu_flat_MES, Value=0.05, DESCRIPTION='maximum element size for flat mumetal'
Variable      Value            Expression
#mu_flat_MES  0.05             0.05
Description: maximum element size for flat mumetal
*** COMMENT ********** / bent mumetal
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#mu_bent_MES, Value=0.03, DESCRIPTION='maximum element size for bent mumetal'
Variable      Value            Expression
#mu_bent_MES  0.03             0.03
Description: maximum element size for bent mumetal
*** COMMENT ********** / air (total potential)
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#air_tot_MES, Value=0.02, DESCRIPTION='maximum element size for total potential air'
Variable      Value            Expression
#air_tot_MES  0.02             0.02
Description: maximum element size for total potential air
*** COMMENT ********** / air (reduced potential)
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#air_red_MES, Value=0.3, DESCRIPTION='maximum element size for reduced potential air'
Variable      Value            Expression
#air_red_MES  0.3              0.3
Description: maximum element size for reduced potential air
**** FILE INPUT ******
*** COMMENT ********** // maximum angle between elements (MAbE)
*** COMMENT ********** / flat mumetal
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#mu_flat_MAbE, Value=30, DESCRIPTION='maximum angle between elements for flat mumetal'
Variable      Value            Expression
#mu_flat_MAbE 30.0             30
Description: maximum angle between elements for flat mumetal
*** COMMENT ********** / bent mumetal
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#mu_bent_MAbE, Value=30, DESCRIPTION='maximum angle between elements for bent mumetal'
Variable      Value            Expression
#mu_bent_MAbE 30.0             30
Description: maximum angle between elements for bent mumetal
*** COMMENT ********** / air
**** FILE INPUT ****** VARIABLE OPTION=PARAMETER, NAME=#air_MAbE, Value=30, DESCRIPTION='maximum angle between elements for air'
Variable      Value            Expression
#air_MAbE     30.0             30
Description: maximum angle between elements for air
**** FILE INPUT ******
**** FILE INPUT ******
*** COMMENT ********** ////////////////////////////////////////////////////////////
*** COMMENT ********** / Modelling
*** COMMENT ********** ////////////////////////////////////////////////////////////
**** FILE INPUT ******
*** COMMENT ********** //// Inner Layer
**** FILE INPUT ******
*** COMMENT ********** / load structural step file
**** FILE INPUT ****** LOAD OPTION=INSERT FILE='stpfiles//inner_layer.STEP'
Imported material name from label: layer1_NAUO1#inner_flat_front_Boss_Extrude2
Imported material name from label: layer1_NAUO2#inner_bracket_Cut_Extrude3
Imported material name from label: layer1_NAUO3#inner_bracket_Cut_Extrude3
Imported material name from label: layer1_NAUO4#inner_bracket_Cut_Extrude3
Imported material name from label: layer1_NAUO5#inner_bracket_Cut_Extrude3
Imported material name from label: layer1_NAUO6#inner_flat_Boss_Extrude2
Imported material name from label: layer1_NAUO7#inner_flat_Boss_Extrude2
Imported material name from label: layer1_NAUO8#inner_bracket_Cut_Extrude3
Imported material name from label: layer1_NAUO9#inner_bracket_Cut_Extrude3
Imported material name from label: layer1_NAUO10#inner_flat_Boss_Extrude2
Imported material name from label: layer1_NAUO11#inner_bracket_Cut_Extrude3
Imported material name from label: layer1_NAUO12#inner_bracket_Cut_Extrude3
Imported material name from label: layer1_NAUO13#inner_bracket_Cut_Extrude3
Imported material name from label: layer1_NAUO14#inner_bracket_Cut_Extrude3
Imported material name from label: layer1_NAUO15#inner_bracket_Cut_Extrude3
Imported material name from label: layer1_NAUO16#inner_bracket_Cut_Extrude3
Imported material name from label: layer1_NAUO17#inner_flat_Boss_Extrude2
Imported material name from label: layer1_NAUO18#inner_flat_Boss_Extrude2
**** FILE INPUT ******
*** COMMENT ********** /// Flat panels
*** COMMENT ********** // front
*** COMMENT ********** / rename body
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-001'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_front' UNIQUENAME='inner_front'
*** COMMENT ********** / set material to air
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_front'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=#air_tot_DSL SIZE=#air_tot_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / face settings
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** / set face to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_front' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_flat_MES NORMALTOL=#mu_flat_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
*** COMMENT ********** // back
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-018'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_back' UNIQUENAME='inner_back'
*** COMMENT ********** / set material to air
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_back'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=#air_tot_DSL SIZE=#air_tot_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / face settings
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** / set face to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_back' IDENTIFIER=A.00010
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_flat_MES NORMALTOL=#mu_flat_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
*** COMMENT ********** // left
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-010'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_left' UNIQUENAME='inner_left'
*** COMMENT ********** / set material to air
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_left'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=#air_tot_DSL SIZE=#air_tot_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / face settings
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** / set face to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_left' IDENTIFIER=A.00010
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_flat_MES NORMALTOL=#mu_flat_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
*** COMMENT ********** // right
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-006'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_right' UNIQUENAME='inner_right'
*** COMMENT ********** / set material to air
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_right'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=#air_tot_DSL SIZE=#air_tot_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / face settings
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** / set face to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_right' IDENTIFIER=A.00010
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_flat_MES NORMALTOL=#mu_flat_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
*** COMMENT ********** // top
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-017'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_top' UNIQUENAME='inner_top'
*** COMMENT ********** / set material to air
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_top'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=#air_tot_DSL SIZE=#air_tot_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / face settings
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** / set face to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_top' IDENTIFIER=A.00010
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_floor' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_flat_MES NORMALTOL=#mu_flat_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
*** COMMENT ********** // bottom
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-007'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_bottom' UNIQUENAME='inner_bottom'
*** COMMENT ********** / set material to air
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_bottom'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=#air_tot_DSL SIZE=#air_tot_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / face settings
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** / set face to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_bottom' IDENTIFIER=A.00010
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_floor' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_flat_MES NORMALTOL=#mu_flat_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ******
*** COMMENT ********** /// brackets
*** COMMENT ********** // front-top
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-005'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_brkt_front_top' UNIQUENAME='inner_brkt_front_top'
*** COMMENT ********** / set material to air
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_brkt_front_top'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=#air_tot_DSL SIZE=#air_tot_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / face settings
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** / set face to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_front_top' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_front_top' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_front_top' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
*** COMMENT ********** // front-bottom
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-003'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_brkt_front_bottom' UNIQUENAME='inner_brkt_front_bottom'
*** COMMENT ********** / set material to air
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_brkt_front_bottom'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=#air_tot_DSL SIZE=#air_tot_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / face settings
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** / set face to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_front_bottom' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_front_bottom' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_front_bottom' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
*** COMMENT ********** // front-left
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-004'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_brkt_front_left' UNIQUENAME='inner_brkt_front_left'
*** COMMENT ********** / set material to air
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_brkt_front_left'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=#air_tot_DSL SIZE=#air_tot_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / face settings
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** / set face to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_front_left' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_front_left' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_front_left' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
*** COMMENT ********** // front-right
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-002'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_brkt_front_right' UNIQUENAME='inner_brkt_front_right'
*** COMMENT ********** / set material to air
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_brkt_front_right'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=#air_tot_DSL SIZE=#air_tot_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / face settings
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** / set face to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_front_right' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_front_right' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_front_right' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
*** COMMENT ********** // bottom-left
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-012'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_brkt_bottom-left' UNIQUENAME='inner_brkt_bottom-left'
*** COMMENT ********** / set material to air
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_brkt_bottom-left'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=#air_tot_DSL SIZE=#air_tot_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / face settings
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** / set face to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_bottom-left' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_floor' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_bottom-left' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_floor' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_bottom-left' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_floor' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
*** COMMENT ********** // bottom-right
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-008'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_brkt_bottom_right' UNIQUENAME='inner_brkt_bottom_right'
*** COMMENT ********** / set material to air
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_brkt_bottom_right'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=#air_tot_DSL SIZE=#air_tot_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / face settings
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** / set face to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_bottom_right' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_floor' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_bottom_right' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_floor' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_bottom_right' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_floor' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
*** COMMENT ********** // top-left
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-011'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_brkt_top_left' UNIQUENAME='inner_brkt_top_left'
*** COMMENT ********** / set material to air
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_brkt_top_left'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=#air_tot_DSL SIZE=#air_tot_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / face settings
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** / set face to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_top_left' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_floor' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_top_left' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_floor' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_top_left' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_floor' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
*** COMMENT ********** // top-right
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-009'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_brkt_top_right' UNIQUENAME='inner_brkt_top_right'
*** COMMENT ********** / set material to air
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_brkt_top_right'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=#air_tot_DSL SIZE=#air_tot_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / face settings
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** / set face to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_top_right' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_floor' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_top_right' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_floor' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_top_right' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_floor' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
*** COMMENT ********** // back-bottom
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-013'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_brkt_back_bottom' UNIQUENAME='inner_brkt_back_bottom'
*** COMMENT ********** / set material to air
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_brkt_back_bottom'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=#air_tot_DSL SIZE=#air_tot_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / face settings
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** / set face to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_back_bottom' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_back_bottom' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_back_bottom' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
*** COMMENT ********** // back-top
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-015'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_brkt_back_top' UNIQUENAME='inner_brkt_back_top'
*** COMMENT ********** / set material to air
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_brkt_back_top'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=#air_tot_DSL SIZE=#air_tot_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / face settings
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** / set face to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_back_top' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_back_top' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_back_top' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
*** COMMENT ********** // back-left
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-016'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_brkt_back_left' UNIQUENAME='inner_brkt_back_left'
*** COMMENT ********** / set material to air
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_brkt_back_left'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=#air_tot_DSL SIZE=#air_tot_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / face settings
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** / set face to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_back_left' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_back_left' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_back_left' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
*** COMMENT ********** // back-right
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-014'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_brkt_back_right' UNIQUENAME='inner_brkt_back_right'
*** COMMENT ********** / set material to air
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_brkt_back_right'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=#air_tot_DSL SIZE=#air_tot_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / face settings
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** / set face to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_back_right' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_back_right' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_back_right' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ******
**** FILE INPUT ******
*** COMMENT ********** // BH data
*** COMMENT ********** / create labels
**** FILE INPUT ****** BHDATA OPTION=NEW  LABEL='innermumetal_door_BH'
**** FILE INPUT ****** BHDATA OPTION=NEW  LABEL='innermumetal_floor_BH'
**** FILE INPUT ******
*** COMMENT ********** / assign bh file to label
**** FILE INPUT ****** BHDATA OPTION=LOAD LABEL=innermumetal_door_BH FILE='bhdata//miniMSR_inner_door_anh.bh'
BHDATA label innermumetal_door_BH loaded with data from the file bhdata//miniMSR_inner_door_anh.bh
**** FILE INPUT ****** BHDATA OPTION=LOAD LABEL=innermumetal_floor_BH FILE='bhdata//miniMSR_inner_floor_anh.bh'
BHDATA label innermumetal_floor_BH loaded with data from the file bhdata//miniMSR_inner_floor_anh.bh
**** FILE INPUT ******
*** COMMENT ********** / setup boundary condition
**** FILE INPUT ****** BOUNDARY UNPICK
**** FILE INPUT ****** BOUNDARY GUIINIT
**** FILE INPUT ****** BOUNDARY PICK 'inner_mumetal_door'
**** FILE INPUT ****** BOUNDARY OPTION=MODIFY CONDITION=THINPLATE PLATEMATERIAL=inner_mumetal_door PLATETHICKNESS=0.0015
**** FILE INPUT ****** BOUNDARY UNPICK
**** FILE INPUT ******
**** FILE INPUT ****** BOUNDARY UNPICK
**** FILE INPUT ****** BOUNDARY GUIINIT
**** FILE INPUT ****** BOUNDARY PICK 'inner_mumetal_floor'
**** FILE INPUT ****** BOUNDARY OPTION=MODIFY CONDITION=THINPLATE PLATEMATERIAL=inner_mumetal_floor PLATETHICKNESS=0.0015
**** FILE INPUT ****** BOUNDARY UNPICK
**** FILE INPUT ******
*** COMMENT ********** / match material label to bh label
**** FILE INPUT ****** MATERIALS UNPICK
**** FILE INPUT ****** MATERIALS GUIINIT
**** FILE INPUT ****** MATERIALS PICK 'inner_mumetal_door'
**** FILE INPUT ****** MATERIALS OPTION=MODIFY MULINEARITY=NONLINEAR MUANISOTROPY=ISOTROPIC BH='innermumetal_door_BH'
**** FILE INPUT ****** MATERIALS UNPICK
**** FILE INPUT ****** MATERIALS UNPICK
**** FILE INPUT ****** MATERIALS GUIINIT
**** FILE INPUT ****** MATERIALS PICK 'inner_mumetal_floor'
**** FILE INPUT ****** MATERIALS OPTION=MODIFY MULINEARITY=NONLINEAR MUANISOTROPY=ISOTROPIC BH='innermumetal_floor_BH'
**** FILE INPUT ****** MATERIALS UNPICK
**** FILE INPUT ******
**** FILE INPUT ******
*** COMMENT ********** //// Outer Layer
**** FILE INPUT ******
*** COMMENT ********** / load structural step file
**** FILE INPUT ****** LOAD OPTION=INSERT FILE='stpfiles//outer_layer.STEP'
Imported material name from label: outer_layer_NAUO1#outer_flat_front_Boss_Extrude2
Imported material name from label: outer_layer_NAUO2#outer_bracket_Cut_Extrude4
Imported material name from label: outer_layer_NAUO3#outer_bracket_Cut_Extrude4
Imported material name from label: outer_layer_NAUO4#outer_flat_Boss_Extrude2
Imported material name from label: outer_layer_NAUO5#outer_flat_Boss_Extrude2
Imported material name from label: outer_layer_NAUO6#outer_bracket_Cut_Extrude4
Imported material name from label: outer_layer_NAUO7#outer_bracket_Cut_Extrude4
Imported material name from label: outer_layer_NAUO8#outer_bracket_Cut_Extrude4
Imported material name from label: outer_layer_NAUO9#outer_bracket_Cut_Extrude4
Imported material name from label: outer_layer_NAUO10#outer_flat_Boss_Extrude2
Imported material name from label: outer_layer_NAUO11#outer_bracket_Cut_Extrude4
Imported material name from label: outer_layer_NAUO12#outer_bracket_Cut_Extrude4
Imported material name from label: outer_layer_NAUO13#outer_bracket_Cut_Extrude4
Imported material name from label: outer_layer_NAUO14#outer_flat_Boss_Extrude2
Imported material name from label: outer_layer_NAUO15#outer_bracket_Cut_Extrude4
Imported material name from label: outer_layer_NAUO16#outer_bracket_Cut_Extrude4
Imported material name from label: outer_layer_NAUO17#outer_bracket_Cut_Extrude4
Imported material name from label: outer_layer_NAUO18#outer_flat_Boss_Extrude2
Imported material name from label: outer_layer_NAUO19#outer_bracket_Cut_Extrude4
Imported material name from label: outer_layer_NAUO20#outer_bracket_Cut_Extrude4
Imported material name from label: outer_layer_NAUO21#outer_bracket_Cut_Extrude4
Imported material name from label: outer_layer_NAUO22#outer_bracket_Cut_Extrude4
Imported material name from label: outer_layer_NAUO23#outer_bracket_Cut_Extrude4
Imported material name from label: outer_layer_NAUO24#outer_bracket_Cut_Extrude4
Imported material name from label: outer_layer_NAUO25#outer_flat_Boss_Extrude2
Imported material name from label: outer_layer_NAUO26#outer_flat_Boss_Extrude2
**** FILE INPUT ******
*** COMMENT ********** /// Flat panels
*** COMMENT ********** // front
*** COMMENT ********** / rename body
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-001'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='outer_front' UNIQUENAME='outer_front'
*** COMMENT ********** / set material to air
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_front'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=#air_tot_DSL SIZE=#air_tot_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / face settings
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** / set face to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_front' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_flat_MES NORMALTOL=#mu_flat_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
*** COMMENT ********** // back
*** COMMENT ********** / rename body
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-026'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='outer_back' UNIQUENAME='outer_back'
*** COMMENT ********** / set material to air
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_back'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=#air_tot_DSL SIZE=#air_tot_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / face settings
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** / set face to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_back' IDENTIFIER=A.00010
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_flat_MES NORMALTOL=#mu_flat_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
*** COMMENT ********** // left
*** COMMENT ********** / rename body
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-014'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='outer_left' UNIQUENAME='outer_left'
*** COMMENT ********** / set material to air
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_left'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=#air_tot_DSL SIZE=#air_tot_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / face settings
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** / set face to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_left' IDENTIFIER=A.00010
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_flat_MES NORMALTOL=#mu_flat_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
*** COMMENT ********** // right
*** COMMENT ********** / rename body
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-018'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='outer_right' UNIQUENAME='outer_right'
*** COMMENT ********** / set material to air
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_right'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=#air_tot_DSL SIZE=#air_tot_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / face settings
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** / set face to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_right' IDENTIFIER=A.00010
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_flat_MES NORMALTOL=#mu_flat_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
*** COMMENT ********** // top
*** COMMENT ********** / rename body
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-025'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='outer_top' UNIQUENAME='outer_top'
*** COMMENT ********** / set material to air
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_top'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=#air_tot_DSL SIZE=#air_tot_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / face settings
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** / set face to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_top' IDENTIFIER=A.00010
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_floor' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_flat_MES NORMALTOL=#mu_flat_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
*** COMMENT ********** // bottom
*** COMMENT ********** / rename body
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-005'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='outer_bottom' UNIQUENAME='outer_bottom'
*** COMMENT ********** / set material to air
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_bottom'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=#air_tot_DSL SIZE=#air_tot_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / face settings
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** / set face to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_bottom' IDENTIFIER=A.00010
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_floor' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_flat_MES NORMALTOL=#mu_flat_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
*** COMMENT ********** /// brackets
*** COMMENT ********** // front-top
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-021'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='outer_brkt_front_top' UNIQUENAME='outer_brkt_front_top'
*** COMMENT ********** / set material to air
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_brkt_front_top'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=#air_tot_DSL SIZE=#air_tot_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / face settings
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** / set face to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_front_top' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_front_top' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_front_top' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
*** COMMENT ********** // front-bottom
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-003'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='outer_brkt_front_bottom' UNIQUENAME='outer_brkt_front_bottom'
*** COMMENT ********** / set material to air
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_brkt_front_bottom'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=#air_tot_DSL SIZE=#air_tot_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / face settings
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** / set face to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_front_bottom' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_front_bottom' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_front_bottom' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
*** COMMENT ********** // front-left
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-013'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='outer_brkt_front_left' UNIQUENAME='outer_brkt_front_left'
*** COMMENT ********** / set material to air
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_brkt_front_left'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=#air_tot_DSL SIZE=#air_tot_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / face settings
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** / set face to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_front_left' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_front_left' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_front_left' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
*** COMMENT ********** // front-right
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-017'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='outer_brkt_front_right' UNIQUENAME='outer_brkt_front_right'
*** COMMENT ********** / set material to air
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_brkt_front_right'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=#air_tot_DSL SIZE=#air_tot_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / face settings
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** / set face to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_front_right' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_front_right' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_front_right' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
*** COMMENT ********** // bottom-left
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-015'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='outer_brkt_bottom-left' UNIQUENAME='outer_brkt_bottom-left'
*** COMMENT ********** / set material to air
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_brkt_bottom-left'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=#air_tot_DSL SIZE=#air_tot_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / face settings
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** / set face to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_bottom-left' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_floor' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_bottom-left' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_floor' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_bottom-left' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_floor' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
*** COMMENT ********** // bottom-right
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-019'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='outer_brkt_bottom_right' UNIQUENAME='outer_brkt_bottom_right'
*** COMMENT ********** / set material to air
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_brkt_bottom_right'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=#air_tot_DSL SIZE=#air_tot_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / face settings
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** / set face to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_bottom_right' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_floor' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_bottom_right' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_floor' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_bottom_right' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_floor' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
*** COMMENT ********** // top-left
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-022'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='outer_brkt_top_left' UNIQUENAME='outer_brkt_top_left'
*** COMMENT ********** / set material to air
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_brkt_top_left'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=#air_tot_DSL SIZE=#air_tot_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / face settings
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** / set face to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_top_left' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_floor' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_top_left' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_floor' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_top_left' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_floor' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
*** COMMENT ********** // top-right
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-024'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='outer_brkt_top_right' UNIQUENAME='outer_brkt_top_right'
*** COMMENT ********** / set material to air
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_brkt_top_right'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=#air_tot_DSL SIZE=#air_tot_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / face settings
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** / set face to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_top_right' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_floor' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_top_right' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_floor' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_top_right' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_floor' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
*** COMMENT ********** // back-bottom
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-020'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='outer_brkt_back_bottom' UNIQUENAME='outer_brkt_back_bottom'
*** COMMENT ********** / set material to air
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_brkt_back_bottom'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=#air_tot_DSL SIZE=#air_tot_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / face settings
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** / set face to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_back_bottom' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_back_bottom' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_back_bottom' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
*** COMMENT ********** // back-top
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-023'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='outer_brkt_back_top' UNIQUENAME='outer_brkt_back_top'
*** COMMENT ********** / set material to air
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_brkt_back_top'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=#air_tot_DSL SIZE=#air_tot_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / face settings
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** / set face to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_back_top' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_back_top' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_back_top' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
*** COMMENT ********** // back-left
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-012'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='outer_brkt_back_left' UNIQUENAME='outer_brkt_back_left'
*** COMMENT ********** / set material to air
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_brkt_back_left'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=#air_tot_DSL SIZE=#air_tot_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / face settings
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** / set face to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_back_left' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_back_left' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_back_left' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
*** COMMENT ********** // back-right
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-016'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='outer_brkt_back_right' UNIQUENAME='outer_brkt_back_right'
*** COMMENT ********** / set material to air
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_brkt_back_right'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=#air_tot_DSL SIZE=#air_tot_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / face settings
**** FILE INPUT ****** FILTER TYPE=FACE
*** COMMENT ********** / set face to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_back_right' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_back_right' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_back_right' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal_door' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ******
*** COMMENT ********** // BH data
*** COMMENT ********** / create label
**** FILE INPUT ****** BHDATA OPTION=NEW  LABEL='outermumetal_door_BH'
**** FILE INPUT ****** BHDATA OPTION=NEW  LABEL='outermumetal_floor_BH'
**** FILE INPUT ******
*** COMMENT ********** / assign bh file to label
**** FILE INPUT ****** BHDATA OPTION=LOAD LABEL=outermumetal_door_BH FILE='bhdata//miniMSR_outer_door_anh.bh'
BHDATA label outermumetal_door_BH loaded with data from the file bhdata//miniMSR_outer_door_anh.bh
**** FILE INPUT ****** BHDATA OPTION=LOAD LABEL=outermumetal_floor_BH FILE='bhdata//miniMSR_outer_floor_anh.bh'
BHDATA label outermumetal_floor_BH loaded with data from the file bhdata//miniMSR_outer_floor_anh.bh
**** FILE INPUT ******
*** COMMENT ********** / setup boundary condition
**** FILE INPUT ****** BOUNDARY UNPICK
**** FILE INPUT ****** BOUNDARY GUIINIT
**** FILE INPUT ****** BOUNDARY PICK 'outer_mumetal_door'
**** FILE INPUT ****** BOUNDARY OPTION=MODIFY CONDITION=THINPLATE PLATEMATERIAL=outer_mumetal_door PLATETHICKNESS=0.002
**** FILE INPUT ****** BOUNDARY UNPICK
**** FILE INPUT ****** BOUNDARY UNPICK
**** FILE INPUT ****** BOUNDARY GUIINIT
**** FILE INPUT ****** BOUNDARY PICK 'outer_mumetal_floor'
**** FILE INPUT ****** BOUNDARY OPTION=MODIFY CONDITION=THINPLATE PLATEMATERIAL=outer_mumetal_floor PLATETHICKNESS=0.002
**** FILE INPUT ****** BOUNDARY UNPICK
**** FILE INPUT ******
*** COMMENT ********** / match material label to bh label
**** FILE INPUT ****** MATERIALS UNPICK
**** FILE INPUT ****** MATERIALS GUIINIT
**** FILE INPUT ****** MATERIALS PICK 'outer_mumetal_door'
**** FILE INPUT ****** MATERIALS OPTION=MODIFY MULINEARITY=NONLINEAR MUANISOTROPY=ISOTROPIC BH='outermumetal_door_BH'
**** FILE INPUT ****** MATERIALS UNPICK
**** FILE INPUT ****** MATERIALS UNPICK
**** FILE INPUT ****** MATERIALS GUIINIT
**** FILE INPUT ****** MATERIALS PICK 'outer_mumetal_floor'
**** FILE INPUT ****** MATERIALS OPTION=MODIFY MULINEARITY=NONLINEAR MUANISOTROPY=ISOTROPIC BH='outermumetal_floor_BH'
**** FILE INPUT ****** MATERIALS UNPICK
**** FILE INPUT ******
**** FILE INPUT ******
*** COMMENT ********** // create total air block
**** FILE INPUT ****** SKETCH OBJECT=BLOCK +COMPLETE
**** FILE INPUT ****** PREVIEW OPTION=END REDISPLAY=NO
**** FILE INPUT ****** BLOCK Name='Block' X0=0.5 Y0=0.5 Z0=0.5 X1=-0.5 Y1=-0.5 Z1=-0.5 MATERIALLABEL='Air' LEVEL=#air_tot_DSL
*** COMMENT ********** / rename
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Block'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='total_air_block' UNIQUENAME='total_air_block'
*** COMMENT ********** / mesh parameters
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='total_air_block'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Total ELEMENTTYPE=Linear LEVEL=#air_tot_DSL SIZE=#air_tot_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
**** FILE INPUT ******
*** COMMENT ********** // create reduced air block
**** FILE INPUT ****** SKETCH OBJECT=BLOCK +COMPLETE
**** FILE INPUT ****** PREVIEW OPTION=END REDISPLAY=NO
**** FILE INPUT ****** BLOCK Name='Block' X0=2 Y0=2 Z0=2 X1=-2 Y1=-2 Z1=-2 MATERIALLABEL='Air' LEVEL=#air_red_DSL
*** COMMENT ********** / rename
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Block'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='reduced_air_block' UNIQUENAME='reduced_air_block'
*** COMMENT ********** / mesh parameters
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='reduced_air_block'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='Air' POTENTIAL=Reduced ELEMENTTYPE=Linear LEVEL=#air_red_DSL SIZE=#air_red_MES NORMALTOL=#air_MAbE ELEMSHAPEPREF=NONE
**** FILE INPUT ******
*** COMMENT ********** / zoom out view
**** FILE INPUT ****** THREED XORIGIN=0 YORIGIN=0 ZORIGIN=0 ROTX=20 ROTY=45 ROTZ=0 XASPECT=1 YASPECT=1 ZASPECT=1 SIZE=5 FACETANGLE=5 PERSPECTIVE=YES LINECOLOUR=YES OPTION=SETVIEW
**** FILE INPUT ******
*** COMMENT ********** //// Analysis Settings
**** FILE INPUT ****** MULTIPHYSICS OPTION=RESET
**** FILE INPUT ****** ANALYSISDATA OPTION=SET PROGRAM=TOSCAMAGN HX=0 HY=0 HZ=50 LINEAR=NO NITERATIONS=21 NLITERTYPE=NEWTON POTENTIALCUT=YES RHS=ADAPTIVE SCALEDRIVE=ALL TOLERANCE=0.001 USEDEFORMEDMESH=NO USEDIRECTSOLVER=NO
**** FILE INPUT ******
*** COMMENT ********** //// Symmetry
*** COMMENT ********** /++ BACKGROUND OPTION=LOAD
*** COMMENT ********** /++ BACKGROUND OPTION=SET SHAPE=BLOCK SCALEX=1 SCALEY=1 SCALEZ=1   XYSYMMETRYPLANE=NO YZSYMMETRYPLANE=YES ROTZNUM=1 ZXSYMMETRYPLANE=YES EMRYZ=TANGMAGN EMRZX=NORMMAGN
**** FILE INPUT ******
**** FILE INPUT ******
*** COMMENT ********** //// Create Model Body
**** FILE INPUT ****** MODEL CREATE
**** FILE INPUT ******
*** COMMENT ********** //// Meshing
*** COMMENT ********** / Surface Mesh
**** FILE INPUT ****** MESH GENERATOR=AUTOMATIC SIZE=0.3 NORMALTOL=30.0 SURFACETOL=0.0 TOLERANCE=1.0E-06 TYPE=PREFERTETRA
Mesh generator Type II
696 faces completed, generating 240752 surface elements

Triangular facet quality on 696 faces
Triangles: 240752, Max: 1.0, Min: 0.151653, Mean: 0.928819, Power2Mean: 0.818205

**** FILE INPUT ******
*** COMMENT ********** / Volume Mesh
**** FILE INPUT ****** FILL TOL=1.0E-06

Tetrahedral element quality on 38 cells
Tetrahedra: 1140580, Max: 1.612384, Min: 0.109007, RMS: 1.017529, Power2Mean: 0.87521, Worst cell: 20

Total nodes:    193893
Tetrahedra:     1140580
Total volume: 64.0
**** FILE INPUT ******
*** COMMENT ********** //// Create Database
**** FILE INPUT ****** SOLVERS SOLVENOW=NO SAVEMODEL=YES,
**** FILE INPUT ****** SOLVERS OPTION=TEST FILE='mini_MSR' UNITS=METRE ELEMENT=MIXED SURFACE=CURVED
**** FILE INPUT ****** COMMENT CLEAR=YES TYPE=DBTITLE
**** FILE INPUT ****** SOLVERS OPTION=OVERWRITE
Opening database mini_MSR.op3, simulation number 0 on 23/May/2024 at 12:44:50
   Magnetostatic Analysis (TOSCA)

   File: mini_MSR.op3 simulation:  1
   Created on: 23/May/2024 12:44:34
   In Directory:
   C:\Users\mzhao\OPERA_miniMSR\
   By Machine:
   Node: UCNENG01. Processor: Intel64 Family 6 Model 42 Stepping 7, GenuineIntel. System: Windows 10 Version 2009. version windows (10.0.19045)
   Log Files: Modeller_202405230944315521.log/lp
   Simulation created using: Opera-3d/Modeller Version 21.3.3
   User did not enter a title

   SI units

   No conductors
   External field vector (0.0,0.0,50.0 A/m)
   Adaptive RHS integrals

Warning: Automatic potential cuts have been disabled because there are thin plates.

   Boundary conditions:
     TANGMAGN
        Note: Assumed as the default boundary condition to one or more external faces.
     THINPLATE
        thin plate of material OUTER_MUMETAL_FLOOR, thickness: 2.0E-03 [m]
        thin plate of material OUTER_MUMETAL_DOOR, thickness: 2.0E-03 [m]
        thin plate of material INNER_MUMETAL_FLOOR, thickness: 1.5E-03 [m]
        thin plate of material INNER_MUMETAL_DOOR, thickness: 1.5E-03 [m]

   Nonlinear iteration data
   Newton-Raphson Iterations
   Maximum Iterations: 21
   Tolerance: 1.0E-03

   No periodicity conditions have been defined

   1 solution scaling factor defined for all drives: 1.0

   Materials defined:
   OUTER_MUMETAL_FLOOR
     Isotropic permeability: bhdata//miniMSR_outer_floor_anh.bh (15193.9 )
   OUTER_MUMETAL_DOOR
     Isotropic permeability: bhdata//miniMSR_outer_door_anh.bh (5933.308)
   INNER_MUMETAL_FLOOR
     Isotropic permeability: bhdata//miniMSR_inner_floor_anh.bh (25009.79)
   INNER_MUMETAL_DOOR
     Isotropic permeability: bhdata//miniMSR_inner_door_anh.bh (15402.39)
   AIR

   809564 nodes in the model
   1696913 edges in the model
   Both linear and quadratic elements exist in the model
   1140580 tetrahedral elements (880220 linear and 260360 quadratic)
   1140580 elements in total

   Symmetry:
   none

WARNINGS:
1 warning when preparing simulation
**** FILE INPUT ******
**** FILE INPUT ******
**** FILE INPUT ******
**** FILE INPUT ******
**** FILE INPUT ******
End of command file.
*** CONTROL COMMAND ** $ END COMI
**** COMMAND INPUT ***
**** COMMAND INPUT *** BHDATA OPTION=VIEW  label=outermumetal_door_BH
**** COMMAND INPUT ***
**** COMMAND INPUT *** BHDATA OPTION=NEW  LABEL='mumetal_PMT'
**** COMMAND INPUT *** BHDATA OPTION=LOAD LABEL=mumetal_PMT FILE=C:/Users/mzhao/OPERA_miniMSR/bhdata/PMTshieldB_anh.bh
BHDATA label mumetal_PMT loaded with data from the file C:/Users/mzhao/OPERA_miniMSR/bhdata/PMTshieldB_anh.bh
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT *** BHDATA OPTION=VIEW  label=innermumetal_floor_BH
**** COMMAND INPUT ***
**** COMMAND INPUT *** MODEL DELETE
**** COMMAND INPUT ***
**** COMMAND INPUT *** BACKGROUND OPTION=LOAD
**** COMMAND INPUT ***
**** COMMAND INPUT *** BACKGROUND OPTION=SET SHAPE=NONE
**** COMMAND INPUT ***
**** COMMAND INPUT *** BACKGROUND OPTION=LOAD
**** COMMAND INPUT ***
**** COMMAND INPUT *** BACKGROUND OPTION=SET SHAPE=BLOCK SCALEX=1 SCALEY=1 SCALEZ=1   XYSYMMETRYPLANE=NO YZSYMMETRYPLANE=YES ROTZNUM=1 ZXSYMMETRYPLANE=YES EMRYZ=TANGMAGN EMRZX=TANGMAGN
**** COMMAND INPUT ***
**** COMMAND INPUT *** MODEL CREATE
**** COMMAND INPUT ***
**** COMMAND INPUT *** MESH GENERATOR=AUTOMATIC SIZE=0.3 NORMALTOL=30.0 SURFACETOL=0.0 TOLERANCE=1.0E-06 TYPE=PREFERTETRA
Mesh generator Type II
282 faces completed, generating 69458 surface elements

Triangular facet quality on 282 faces
Triangles: 69458, Max: 1.0, Min: 0.151653, Mean: 0.925969, Power2Mean: 0.791242

**** COMMAND INPUT ***
**** COMMAND INPUT *** FILL TOL=1.0E-06

Tetrahedral element quality on 20 cells
Tetrahedra: 437483, Max: 1.579702, Min: 0.145365, RMS: 1.016333, Power2Mean: 0.913488, Worst cell: 11

Total nodes:    75674
Tetrahedra:     437483
Total volume: 16.0
**** COMMAND INPUT ***
**** COMMAND INPUT *** MULTIPHYSICS OPTION=RESET
**** COMMAND INPUT ***
**** COMMAND INPUT *** DBCASEDATA PROGRAM=TOSCAMAGN OPTION=CLEAR
1 cases have been deleted.
**** COMMAND INPUT ***
**** COMMAND INPUT *** ANALYSISDATA OPTION=SET PROGRAM=TOSCAMAGN HX=0 HY=50 HZ=0 LINEAR=NO NITERATIONS=100 NLITERTYPE=NEWTON POTENTIALCUT=YES RHS=ADAPTIVE SCALEDRIVE=ALL TOLERANCE=0.001 USEDEFORMEDMESH=NO USEDIRECTSOLVER=NO
**** COMMAND INPUT ***
**** COMMAND INPUT *** MULTIPHYSICS OPTION=RESET
**** COMMAND INPUT ***
**** COMMAND INPUT *** DBCASEDATA PROGRAM=TOSCAMAGN OPTION=CLEAR
1 cases have been deleted.
**** COMMAND INPUT ***
**** COMMAND INPUT *** ANALYSISDATA OPTION=SET PROGRAM=TOSCAMAGN HX=0 HY=50 HZ=0 LINEAR=NO NITERATIONS=100 NLITERTYPE=NEWTON POTENTIALCUT=YES RHS=ADAPTIVE SCALEDRIVE=ALL TOLERANCE=0.001 USEDEFORMEDMESH=NO USEDIRECTSOLVER=NO
**** COMMAND INPUT ***
**** COMMAND INPUT *** SOLVERS SOLVENOW=NO SAVEMODEL=YES,
**** COMMAND INPUT *** SOLVERS OPTION=TEST FILE='mini_MSR.op3' UNITS=METRE ELEMENT=MIXED SURFACE=CURVED
**** COMMAND INPUT *** COMMENT CLEAR=YES TYPE=DBTITLE
**** COMMAND INPUT *** SOLVERS OPTION=OVERWRITE
Opening database mini_MSR.op3, simulation number 0 on 24/May/2024 at 09:07:26
   Magnetostatic Analysis (TOSCA)

   File: mini_MSR.op3 simulation:  1
   Created on: 24/May/2024 09:07:22
   In Directory:
   C:\Users\mzhao\OPERA_miniMSR\
   By Machine:
   Node: UCNENG01. Processor: Intel64 Family 6 Model 42 Stepping 7, GenuineIntel. System: Windows 10 Version 2009. version windows (10.0.19045)
   Log Files: Modeller_202405230944315521.log/lp
   Simulation created using: Opera-3d/Modeller Version 21.3.3
   User did not enter a title

   SI units

   No conductors
   External field vector (0.0,50.0,0.0 A/m)
   Adaptive RHS integrals

Warning: Automatic potential cuts have been disabled because there are thin plates.

   Boundary conditions:
     TANGMAGN
        Note: Assumed as the default boundary condition to one or more external faces.
     THINPLATE
        thin plate of material OUTER_MUMETAL_DOOR, thickness: 2.0E-03 [m]
        thin plate of material OUTER_MUMETAL_FLOOR, thickness: 2.0E-03 [m]
        thin plate of material INNER_MUMETAL_DOOR, thickness: 1.5E-03 [m]
        thin plate of material INNER_MUMETAL_FLOOR, thickness: 1.5E-03 [m]

   Nonlinear iteration data
   Newton-Raphson Iterations
   Maximum Iterations: 100
   Tolerance: 1.0E-03

   No periodicity conditions have been defined

   1 solution scaling factor defined for all drives: 1.0

   Materials defined:
   OUTER_MUMETAL_DOOR
     Isotropic permeability: bhdata//miniMSR_outer_door_anh.bh (5933.308)
   OUTER_MUMETAL_FLOOR
     Isotropic permeability: bhdata//miniMSR_outer_floor_anh.bh (15193.9 )
   INNER_MUMETAL_DOOR
     Isotropic permeability: bhdata//miniMSR_inner_door_anh.bh (15402.39)
   INNER_MUMETAL_FLOOR
     Isotropic permeability: bhdata//miniMSR_inner_floor_anh.bh (25009.79)
   AIR

   233778 nodes in the model
   608545 edges in the model
   Both linear and quadratic elements exist in the model
   437483 tetrahedral elements (370523 linear and 66960 quadratic)
   437483 elements in total

   Symmetry:
   Positive reflection in YZ plane
   Positive reflection in ZX plane

WARNINGS:
1 warning when preparing simulation
**** COMMAND INPUT ***
**** COMMAND INPUT *** MODEL DELETE
**** COMMAND INPUT ***
**** COMMAND INPUT *** BACKGROUND OPTION=LOAD
**** COMMAND INPUT ***
**** COMMAND INPUT *** BACKGROUND OPTION=SET SHAPE=BLOCK SCALEX=2 SCALEY=2 SCALEZ=2   XYSYMMETRYPLANE=NO YZSYMMETRYPLANE=YES ROTZNUM=1 ZXSYMMETRYPLANE=YES EMRYZ=TANGMAGN EMRZX=TANGMAGN
**** COMMAND INPUT ***
**** COMMAND INPUT *** MODEL CREATE
**** COMMAND INPUT ***
**** COMMAND INPUT *** MESH GENERATOR=AUTOMATIC SIZE=0.3 NORMALTOL=30.0 SURFACETOL=0.0 TOLERANCE=1.0E-06 TYPE=PREFERTETRA
Mesh generator Type II
288 faces completed, generating 73467 surface elements

Triangular facet quality on 288 faces
Triangles: 73467, Max: 1.0, Min: 0.152163, Mean: 0.928316, Power2Mean: 0.79885

**** COMMAND INPUT ***
**** COMMAND INPUT *** FILL TOL=1.0E-06

Tetrahedral element quality on 21 cells
Tetrahedra: 465896, Max: 1.660083, Min: 0.104301, RMS: 1.017269, Power2Mean: 0.917744, Worst cell: 7

Total nodes:    81257
Tetrahedra:     465896
Total volume: 128.0
**** COMMAND INPUT ***
**** COMMAND INPUT *** SOLVERS SOLVENOW=YES SAVEMODEL=YES,
**** COMMAND INPUT *** SOLVERS OPTION=TEST FILE='mini_MSR.op3' UNITS=METRE ELEMENT=MIXED SURFACE=CURVED
**** COMMAND INPUT *** COMMENT CLEAR=YES TYPE=DBTITLE
**** COMMAND INPUT *** SOLVERS OPTION=OVERWRITE
Opening database mini_MSR.op3, simulation number 0 on 24/May/2024 at 09:24:51
   Magnetostatic Analysis (TOSCA)

   File: mini_MSR.op3 simulation:  1
   Created on: 24/May/2024 09:24:47
   In Directory:
   C:\Users\mzhao\OPERA_miniMSR\
   By Machine:
   Node: UCNENG01. Processor: Intel64 Family 6 Model 42 Stepping 7, GenuineIntel. System: Windows 10 Version 2009. version windows (10.0.19045)
   Log Files: Modeller_202405230944315521.log/lp
   Simulation created using: Opera-3d/Modeller Version 21.3.3
   User did not enter a title

   SI units

   No conductors
   External field vector (0.0,50.0,0.0 A/m)
   Adaptive RHS integrals

Warning: Automatic potential cuts have been disabled because there are thin plates.

   Boundary conditions:
     TANGMAGN
        Note: Assumed as the default boundary condition to one or more external faces.
     THINPLATE
        thin plate of material OUTER_MUMETAL_DOOR, thickness: 2.0E-03 [m]
        thin plate of material OUTER_MUMETAL_FLOOR, thickness: 2.0E-03 [m]
        thin plate of material INNER_MUMETAL_DOOR, thickness: 1.5E-03 [m]
        thin plate of material INNER_MUMETAL_FLOOR, thickness: 1.5E-03 [m]

   Nonlinear iteration data
   Newton-Raphson Iterations
   Maximum Iterations: 100
   Tolerance: 1.0E-03

   No periodicity conditions have been defined

   1 solution scaling factor defined for all drives: 1.0

   Materials defined:
   OUTER_MUMETAL_DOOR
     Isotropic permeability: bhdata//miniMSR_outer_door_anh.bh (5933.308)
   OUTER_MUMETAL_FLOOR
     Isotropic permeability: bhdata//miniMSR_outer_floor_anh.bh (15193.9 )
   INNER_MUMETAL_DOOR
     Isotropic permeability: bhdata//miniMSR_inner_door_anh.bh (15402.39)
   INNER_MUMETAL_FLOOR
     Isotropic permeability: bhdata//miniMSR_inner_floor_anh.bh (25009.79)
   AIR

   239580 nodes in the model
   645284 edges in the model
   Both linear and quadratic elements exist in the model
   465896 tetrahedral elements (399048 linear and 66848 quadratic)
   465896 elements in total

   Symmetry:
   Positive reflection in YZ plane
   Positive reflection in ZX plane

WARNINGS:
1 warning when preparing simulation

Do you want to continue?
YES or NO
*** LOOP COMMAND ***** yes
*** CONTROL COMMAND ** $OS OPERAANL WRD2=ToscaMagn WRD3='mini_MSR.op3' WRD4=BACK
**** COMMAND INPUT ***
qWarning :A non fatal error has happened, please inform Opera FEA Support
Retrying to obtain clipboard.
**** COMMAND INPUT *** MODEL DELETE
**** COMMAND INPUT ***
**** COMMAND INPUT *** BACKGROUND OPTION=LOAD
**** COMMAND INPUT ***
**** COMMAND INPUT *** BACKGROUND OPTION=SET SHAPE=BLOCK SCALEX=0.5 SCALEY=0.5 SCALEZ=0.5   XYSYMMETRYPLANE=NO YZSYMMETRYPLANE=YES ROTZNUM=1 ZXSYMMETRYPLANE=YES EMRYZ=TANGMAGN EMRZX=TANGMAGN
Background shape scales must be greater than or equal to 1
The model symmetry information will not be changed.
**** COMMAND INPUT ***
**** COMMAND INPUT *** BACKGROUND OPTION=SET SHAPE=BLOCK SCALEX=5 SCALEY=5 SCALEZ=5   XYSYMMETRYPLANE=NO YZSYMMETRYPLANE=YES ROTZNUM=1 ZXSYMMETRYPLANE=YES EMRYZ=TANGMAGN EMRZX=TANGMAGN
**** COMMAND INPUT ***
**** COMMAND INPUT *** MODEL CREATE
**** COMMAND INPUT ***
**** COMMAND INPUT *** MESH GENERATOR=AUTOMATIC SIZE=0.3 NORMALTOL=30.0 SURFACETOL=0.0 TOLERANCE=1.0E-06 TYPE=PREFERTETRA
Mesh generator Type II
288 faces completed, generating 96551 surface elements

Triangular facet quality on 288 faces
Triangles: 96551, Max: 1.0, Min: 0.160165, Mean: 0.939164, Power2Mean: 0.831731

**** COMMAND INPUT ***
**** COMMAND INPUT *** FILL TOL=1.0E-06

Tetrahedral element quality on 21 cells
Tetrahedra: 995941, Max: 1.616557, Min: 0.0984774, RMS: 1.019816, Power2Mean: 0.956958, Worst cell: 7

Total nodes:    174325
Tetrahedra:     995941
Total volume: 2000.0
**** COMMAND INPUT ***
**** COMMAND INPUT *** SOLVERS SOLVENOW=YES SAVEMODEL=YES,
**** COMMAND INPUT *** SOLVERS OPTION=TEST FILE='mini_MSR.op3' UNITS=METRE ELEMENT=MIXED SURFACE=CURVED
**** COMMAND INPUT *** COMMENT CLEAR=YES TYPE=DBTITLE
**** COMMAND INPUT *** SOLVERS OPTION=OVERWRITE
Opening database mini_MSR.op3, simulation number 0 on 24/May/2024 at 09:44:29
   Magnetostatic Analysis (TOSCA)

   File: mini_MSR.op3 simulation:  1
   Created on: 24/May/2024 09:44:21
   In Directory:
   C:\Users\mzhao\OPERA_miniMSR\
   By Machine:
   Node: UCNENG01. Processor: Intel64 Family 6 Model 42 Stepping 7, GenuineIntel. System: Windows 10 Version 2009. version windows (10.0.19045)
   Log Files: Modeller_202405230944315521.log/lp
   Simulation created using: Opera-3d/Modeller Version 21.3.3
   User did not enter a title

   SI units

   No conductors
   External field vector (0.0,50.0,0.0 A/m)
   Adaptive RHS integrals

Warning: Automatic potential cuts have been disabled because there are thin plates.

   Boundary conditions:
     TANGMAGN
        Note: Assumed as the default boundary condition to one or more external faces.
     THINPLATE
        thin plate of material OUTER_MUMETAL_DOOR, thickness: 2.0E-03 [m]
        thin plate of material OUTER_MUMETAL_FLOOR, thickness: 2.0E-03 [m]
        thin plate of material INNER_MUMETAL_DOOR, thickness: 1.5E-03 [m]
        thin plate of material INNER_MUMETAL_FLOOR, thickness: 1.5E-03 [m]

   Nonlinear iteration data
   Newton-Raphson Iterations
   Maximum Iterations: 100
   Tolerance: 1.0E-03

   No periodicity conditions have been defined

   1 solution scaling factor defined for all drives: 1.0

   Materials defined:
   OUTER_MUMETAL_DOOR
     Isotropic permeability: bhdata//miniMSR_outer_door_anh.bh (5933.308)
   OUTER_MUMETAL_FLOOR
     Isotropic permeability: bhdata//miniMSR_outer_floor_anh.bh (15193.9 )
   INNER_MUMETAL_DOOR
     Isotropic permeability: bhdata//miniMSR_inner_door_anh.bh (15402.39)
   INNER_MUMETAL_FLOOR
     Isotropic permeability: bhdata//miniMSR_inner_floor_anh.bh (25009.79)
   AIR

   332996 nodes in the model
   1280037 edges in the model
   Both linear and quadratic elements exist in the model
   995941 tetrahedral elements (928893 linear and 67048 quadratic)
   995941 elements in total

   Symmetry:
   Positive reflection in YZ plane
   Positive reflection in ZX plane

WARNINGS:
1 warning when preparing simulation

Do you want to continue?
YES or NO
*** LOOP COMMAND ***** yes
*** CONTROL COMMAND ** $OS OPERAANL WRD2=ToscaMagn WRD3='mini_MSR.op3' WRD4=BACK
**** COMMAND INPUT ***
qWarning :A non fatal error has happened, please inform Opera FEA Support
Retrying to obtain clipboard.
**** COMMAND INPUT *** MODEL DELETE
**** COMMAND INPUT ***
**** COMMAND INPUT *** MODEL CREATE
**** COMMAND INPUT ***
**** COMMAND INPUT *** MESH GENERATOR=AUTOMATIC SIZE=0.1 NORMALTOL=30.0 SURFACETOL=0.0 TOLERANCE=1.0E-06 TYPE=PREFERTETRA
Mesh generator Type II
288 faces completed, generating 317918 surface elements

Triangular facet quality on 288 faces
Triangles: 317918, Max: 1.0, Min: 0.160165, Mean: 0.967461, Power2Mean: 0.926335

**** COMMAND INPUT ***
**** COMMAND INPUT *** FILL TOL=1.0E-06

Tetrahedral element quality on 21 cells
Tetrahedra: 15570162, Max: 1.563216, Min: 0.120858, RMS: 1.021961, Power2Mean: 0.993362, Worst cell: 10

Total nodes:    2627467
Tetrahedra:     15570162
Total volume: 2000.0
**** COMMAND INPUT ***
qWarning :A non fatal error has happened, please inform Opera FEA Support
Retrying to obtain clipboard.
**** COMMAND INPUT *** SOLVERS SOLVENOW=YES SAVEMODEL=YES,
**** COMMAND INPUT *** SOLVERS OPTION=TEST FILE='mini_MSR.op3' UNITS=METRE ELEMENT=MIXED SURFACE=CURVED
**** COMMAND INPUT *** COMMENT CLEAR=YES TYPE=DBTITLE
**** COMMAND INPUT *** SOLVERS OPTION=OVERWRITE
Opening database mini_MSR.op3, simulation number 0 on 24/May/2024 at 10:51:45
   Magnetostatic Analysis (TOSCA)

   File: mini_MSR.op3 simulation:  1
   Created on: 24/May/2024 10:50:27
   In Directory:
   C:\Users\mzhao\OPERA_miniMSR\
   By Machine:
   Node: UCNENG01. Processor: Intel64 Family 6 Model 42 Stepping 7, GenuineIntel. System: Windows 10 Version 2009. version windows (10.0.19045)
   Log Files: Modeller_202405230944315521.log/lp
   Simulation created using: Opera-3d/Modeller Version 21.3.3
   User did not enter a title

   SI units

   No conductors
   External field vector (0.0,50.0,0.0 A/m)
   Adaptive RHS integrals

Warning: Automatic potential cuts have been disabled because there are thin plates.

   Boundary conditions:
     TANGMAGN
        Note: Assumed as the default boundary condition to one or more external faces.
     THINPLATE
        thin plate of material OUTER_MUMETAL_DOOR, thickness: 2.0E-03 [m]
        thin plate of material OUTER_MUMETAL_FLOOR, thickness: 2.0E-03 [m]
        thin plate of material INNER_MUMETAL_DOOR, thickness: 1.5E-03 [m]
        thin plate of material INNER_MUMETAL_FLOOR, thickness: 1.5E-03 [m]

   Nonlinear iteration data
   Newton-Raphson Iterations
   Maximum Iterations: 100
   Tolerance: 1.0E-03

   No periodicity conditions have been defined

   1 solution scaling factor defined for all drives: 1.0

   Materials defined:
   OUTER_MUMETAL_DOOR
     Isotropic permeability: bhdata//miniMSR_outer_door_anh.bh (5933.308)
   OUTER_MUMETAL_FLOOR
     Isotropic permeability: bhdata//miniMSR_outer_floor_anh.bh (15193.9 )
   INNER_MUMETAL_DOOR
     Isotropic permeability: bhdata//miniMSR_inner_door_anh.bh (15402.39)
   INNER_MUMETAL_FLOOR
     Isotropic permeability: bhdata//miniMSR_inner_floor_anh.bh (25009.79)
   AIR

   2789434 nodes in the model
   18423314 edges in the model
   Both linear and quadratic elements exist in the model
   15570162 tetrahedral elements (15502776 linear and 67386 quadratic)
   15570162 elements in total

   Symmetry:
   Positive reflection in YZ plane
   Positive reflection in ZX plane

WARNINGS:
1 warning when preparing simulation

Do you want to continue?
YES or NO
