Opera-3d Modeller                       started on 17/May/2024 at 15:31:56
Opera 2020 SP3 [Build 2]
(c) Copyright Dassault Systemes Ltd 1984-2019. All Rights Reserved.
Node: TUCAN-MAGPC. Processor: Intel64. System: Windows 10
Command language initialization.
Information: modeller.comi was not found in the local or home directory
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT ***
*** CONTROL COMMAND ** $COMI MODE=CONTINUOUS FILE='C:\Users\mzhao\OPERA_miniMSR\test.comi'
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
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_flat_MES NORMALTOL=#mu_flat_MAbE FORMETHOD=NONE BACKMETHOD=NONE
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
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_flat_MES NORMALTOL=#mu_flat_MAbE FORMETHOD=NONE BACKMETHOD=NONE
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
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_flat_MES NORMALTOL=#mu_flat_MAbE FORMETHOD=NONE BACKMETHOD=NONE
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
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_flat_MES NORMALTOL=#mu_flat_MAbE FORMETHOD=NONE BACKMETHOD=NONE
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
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_flat_MES NORMALTOL=#mu_flat_MAbE FORMETHOD=NONE BACKMETHOD=NONE
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
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_flat_MES NORMALTOL=#mu_flat_MAbE FORMETHOD=NONE BACKMETHOD=NONE
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
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_front_top' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_front_top' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
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
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_front_bottom' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_front_bottom' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
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
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_front_left' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_front_left' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
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
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_front_right' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_front_right' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
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
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_bottom-left' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_bottom-left' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
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
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_bottom_right' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_bottom_right' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
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
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_top_left' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_top_left' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
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
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_top_right' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_top_right' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
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
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_back_bottom' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_back_bottom' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
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
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_back_top' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_back_top' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
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
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_back_left' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_back_left' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
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
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_back_right' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_back_right' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='inner_mumetal' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ******
**** FILE INPUT ******
*** COMMENT ********** // BH data
*** COMMENT ********** / create label
**** FILE INPUT ****** BHDATA OPTION=NEW  LABEL='innermumetalBH'
**** FILE INPUT ******
*** COMMENT ********** / assign bh file to label
**** FILE INPUT ****** BHDATA OPTION=LOAD LABEL=innermumetalBH FILE='bhdata//PMTshieldB_anh.bh'
BHDATA label innermumetalBH loaded with data from the file bhdata//PMTshieldB_anh.bh
**** FILE INPUT ******
*** COMMENT ********** / setup boundary condition
**** FILE INPUT ****** BOUNDARY UNPICK
**** FILE INPUT ****** BOUNDARY GUIINIT
**** FILE INPUT ****** BOUNDARY PICK 'inner_mumetal'
**** FILE INPUT ****** BOUNDARY OPTION=MODIFY CONDITION=THINPLATE PLATEMATERIAL=inner_mumetal PLATETHICKNESS=0.0015
**** FILE INPUT ****** BOUNDARY UNPICK
**** FILE INPUT ******
*** COMMENT ********** / match material label to bh label
**** FILE INPUT ****** MATERIALS UNPICK
**** FILE INPUT ****** MATERIALS GUIINIT
**** FILE INPUT ****** MATERIALS PICK 'inner_mumetal'
**** FILE INPUT ****** MATERIALS OPTION=MODIFY MULINEARITY=NONLINEAR MUANISOTROPY=ISOTROPIC BH='innermumetalBH'
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
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_flat_MES NORMALTOL=#mu_flat_MAbE FORMETHOD=NONE BACKMETHOD=NONE
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
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_flat_MES NORMALTOL=#mu_flat_MAbE FORMETHOD=NONE BACKMETHOD=NONE
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
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_flat_MES NORMALTOL=#mu_flat_MAbE FORMETHOD=NONE BACKMETHOD=NONE
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
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_flat_MES NORMALTOL=#mu_flat_MAbE FORMETHOD=NONE BACKMETHOD=NONE
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
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_flat_MES NORMALTOL=#mu_flat_MAbE FORMETHOD=NONE BACKMETHOD=NONE
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
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_flat_MES NORMALTOL=#mu_flat_MAbE FORMETHOD=NONE BACKMETHOD=NONE
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
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_front_top' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_front_top' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
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
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_front_bottom' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_front_bottom' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
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
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_front_left' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_front_left' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
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
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_front_right' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_front_right' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
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
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_bottom-left' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_bottom-left' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
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
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_bottom_right' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_bottom_right' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
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
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_top_left' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_top_left' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
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
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_top_right' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_top_right' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
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
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_back_bottom' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_back_bottom' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
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
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_back_top' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_back_top' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
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
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_back_left' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_back_left' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
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
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_back_right' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_back_right' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY BOUNDARYLABEL='outer_mumetal' LEVEL=#mu_DSL ELEMENTTYPE=Linear SIZE=#mu_bent_MES NORMALTOL=#mu_bent_MAbE FORMETHOD=NONE BACKMETHOD=NONE
**** FILE INPUT ******
**** FILE INPUT ******
*** COMMENT ********** // BH data
*** COMMENT ********** / create label
**** FILE INPUT ****** BHDATA OPTION=NEW  LABEL='outermumetalBH'
**** FILE INPUT ******
*** COMMENT ********** / assign bh file to label
**** FILE INPUT ****** BHDATA OPTION=LOAD LABEL=outermumetalBH FILE='bhdata//PMTshieldB_anh.bh'
BHDATA label outermumetalBH loaded with data from the file bhdata//PMTshieldB_anh.bh
**** FILE INPUT ******
*** COMMENT ********** / setup boundary condition
**** FILE INPUT ****** BOUNDARY UNPICK
**** FILE INPUT ****** BOUNDARY GUIINIT
**** FILE INPUT ****** BOUNDARY PICK 'outer_mumetal'
**** FILE INPUT ****** BOUNDARY OPTION=MODIFY CONDITION=THINPLATE PLATEMATERIAL=outer_mumetal PLATETHICKNESS=0.002
**** FILE INPUT ****** BOUNDARY UNPICK
**** FILE INPUT ******
*** COMMENT ********** / match material label to bh label
**** FILE INPUT ****** MATERIALS UNPICK
**** FILE INPUT ****** MATERIALS GUIINIT
**** FILE INPUT ****** MATERIALS PICK 'outer_mumetal'
**** FILE INPUT ****** MATERIALS OPTION=MODIFY MULINEARITY=NONLINEAR MUANISOTROPY=ISOTROPIC BH='outermumetalBH'
**** FILE INPUT ****** MATERIALS UNPICK
**** FILE INPUT ******
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
**** FILE INPUT ****** ANALYSISDATA OPTION=SET PROGRAM=TOSCAMAGN HX=0 HY=50 HZ=0 LINEAR=NO NITERATIONS=21 NLITERTYPE=NEWTON POTENTIALCUT=YES RHS=ADAPTIVE SCALEDRIVE=ALL TOLERANCE=0.001 USEDEFORMEDMESH=NO USEDIRECTSOLVER=NO
**** FILE INPUT ******
*** COMMENT ********** //// Symmetry
**** FILE INPUT ****** BACKGROUND OPTION=LOAD
**** FILE INPUT ****** BACKGROUND OPTION=SET SHAPE=BLOCK SCALEX=1 SCALEY=1 SCALEZ=1   XYSYMMETRYPLANE=NO YZSYMMETRYPLANE=YES ROTZNUM=1 ZXSYMMETRYPLANE=YES EMRYZ=TANGMAGN EMRZX=NORMMAGN
**** FILE INPUT ******
**** FILE INPUT ******
*** COMMENT ********** //// Create Model Body
**** FILE INPUT ****** MODEL CREATE
**** FILE INPUT ******
*** COMMENT ********** //// Meshing
*** COMMENT ********** / Surface Mesh
**** FILE INPUT ****** MESH GENERATOR=AUTOMATIC SIZE=0.3 NORMALTOL=30.0 SURFACETOL=0.0 TOLERANCE=1.0E-06 TYPE=PREFERTETRA
Mesh generator Type II
282 faces completed, generating 69462 surface elements

Triangular facet quality on 282 faces
Triangles: 69462, Max: 1.0, Min: 0.151653, Mean: 0.92576, Power2Mean: 0.791204

**** FILE INPUT ******
*** COMMENT ********** / Volume Mesh
**** FILE INPUT ****** FILL TOL=1.0E-06

Tetrahedral element quality on 20 cells
Tetrahedra: 434817, Max: 1.576248, Min: 0.132768, RMS: 1.016582, Power2Mean: 0.913547, Worst cell: 16

Total nodes:    75269
Tetrahedra:     434817
Total volume: 16.0
**** FILE INPUT ******
*** COMMENT ********** //// Create Database
**** FILE INPUT ****** SOLVERS SOLVENOW=NO SAVEMODEL=YES,
**** FILE INPUT ****** SOLVERS OPTION=TEST FILE='mini_MSR' UNITS=METRE ELEMENT=MIXED SURFACE=CURVED
**** FILE INPUT ****** COMMENT CLEAR=YES TYPE=DBTITLE
**** FILE INPUT ****** SOLVERS OPTION=OVERWRITE
Opening database mini_MSR.op3, simulation number 0 on 17/May/2024 at 15:32:58
   Magnetostatic Analysis (TOSCA)

   File: mini_MSR.op3 simulation:  1
   Created on: 17/May/2024 15:32:56
   In Directory:
   C:\Users\mzhao\OPERA_miniMSR\
   By Machine:
   Node: TUCAN-MAGPC. Processor: Intel64. System: Windows. version 10
   Log Files: Modeller_202405171531561401.log/lp
   Simulation created using: Opera-3d/Modeller Version 20.3.2
   User did not enter a title

   SI units

   No conductors
   External field vector (0.0,50.0,0.0 A/m)
   Adaptive RHS integrals

Warning: Automatic potential cuts have been disabled because there are thin plates.

   Boundary conditions:
     NORMMAGN
     TANGMAGN
        Note: Assumed as the default boundary condition to one or more external faces.
     THINPLATE
        thin plate of material OUTER_MUMETAL, thickness: 2.0E-03 [m]
        thin plate of material INNER_MUMETAL, thickness: 1.5E-03 [m]

   Nonlinear iteration data
   Newton-Raphson Iterations
   Maximum Iterations: 21
   Tolerance: 1.0E-03

   No periodicity conditions have been defined

   1 solution scaling factor defined for all drives: 1.0

   Materials defined:
   OUTER_MUMETAL
     Isotropic permeability: bhdata//PMTshieldB_anh.bh (72608.37)
   INNER_MUMETAL
     Isotropic permeability: bhdata//PMTshieldB_anh.bh (72608.37)
   AIR

   233000 nodes in the model
   605440 edges in the model
   Both linear and quadratic elements exist in the model
   434817 tetrahedral elements (368106 linear and 66711 quadratic)
   434817 elements in total

   Symmetry:
   Positive reflection in YZ plane
   Negative reflection in ZX plane

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
Opera-3d Modeller                      complete on 17/May/2024 at 15:41:09
                                                00:09:13.09 (00:00:54.43 cp)
