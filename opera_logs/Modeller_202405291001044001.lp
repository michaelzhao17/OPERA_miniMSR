Opera-3d Modeller                       started on 29/May/2024 at 10:01:04
Opera 2021 SP3 [Build 3]
(c) Copyright Dassault Systemes Ltd 1984-2021. All Rights Reserved.
Node: UCNENG01. Processor: Intel64 Family 6 Model 42 Stepping 7, GenuineIntel. System: Windows 10 Version 2009 windows (10.0.19045)
Command language initialization.
Information: modeller.comi was not found in the local or home directory
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT ***
*** CONTROL COMMAND ** $ COMI 'C:\Users\mzhao\OPERA_miniMSR\mesh_test.comi'
Opening file for input: C:\Users\mzhao\OPERA_miniMSR\mesh_test.comi
*** COMMENT ********** // Meshing Troubleshooting
*** COMMENT ********** // May 2024
*** COMMENT ********** // Start with only the bracket.
**** FILE INPUT ******
*** CONTROL COMMAND ** $string yesorno YES
 yesorno='YES'
**** FILE INPUT ****** CLEAR REVERT=NO
Command language initialization.
**** FILE INPUT ****** THREED XORIGIN=0 YORIGIN=0 ZORIGIN=0 ROTX=0 ROTY=0 ROTZ=0 XASPECT=1 YASPECT=1 ZASPECT=1 SIZE=0 FACETANGLE=5 PERSPECTIVE=YES LINECOLOUR=YES OPTION=SETVIEW
**** FILE INPUT ******
*** COMMENT ********** / load structural step file
**** FILE INPUT ****** LOAD OPTION=INSERT FILE='stpfiles//inner_bracket_flat.STEP'
Imported material name from label: inner_bracket_flat_NAUO1#inner_bracket_Cut_Extrude3
Imported material name from label: inner_bracket_flat_NAUO2#inner_flat_Boss_Extrude2
**** FILE INPUT ******
**** FILE INPUT ******
**** FILE INPUT ******
**** FILE INPUT ******
**** FILE INPUT ******
**** FILE INPUT ******
**** FILE INPUT ******
**** FILE INPUT ******
**** FILE INPUT ******
**** FILE INPUT ******
**** FILE INPUT ******
**** FILE INPUT ******
**** FILE INPUT ******
**** FILE INPUT ******
**** FILE INPUT ******
**** FILE INPUT ******
**** FILE INPUT ******
**** FILE INPUT ******
**** FILE INPUT ******
**** FILE INPUT ******
**** FILE INPUT ******
**** FILE INPUT ******
End of command file.
*** CONTROL COMMAND ** $ END COMI
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=UNPICK REDISPLAY=YES
**** COMMAND INPUT ***
**** COMMAND INPUT *** FILTER TYPE=FACE
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=PICK
**** COMMAND INPUT *** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='Unnamed:AAA-002' IDENTIFIER=A.00010
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** COMMAND INPUT *** FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=2 BACKOFFSET=0.75
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=PICK
**** COMMAND INPUT *** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='Unnamed:AAA-001' IDENTIFIER=A.00006
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** COMMAND INPUT *** FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=2 BACKOFFSET=0.75
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=PICK
**** COMMAND INPUT *** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='Unnamed:AAA-001' IDENTIFIER=A.00013
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** COMMAND INPUT *** FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=2 BACKOFFSET=0.75
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=PICK
**** COMMAND INPUT *** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='Unnamed:AAA-001' IDENTIFIER=A.00018
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** COMMAND INPUT *** FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=2 BACKOFFSET=0.75
**** COMMAND INPUT ***
**** COMMAND INPUT *** MODEL CREATE
**** COMMAND INPUT ***
**** COMMAND INPUT *** MESH GENERATOR=AUTOMATIC SIZE=10.0 NORMALTOL=30.0 SURFACETOL=0.0 TOLERANCE=1.0E-06 TYPE=PREFERTETRA
Mesh generator Type II
72 faces completed, generating 36884 surface elements

Triangular facet quality on 72 faces
Triangles: 36884, Max: 0.999999, Min: 0.167905, Mean: 0.937969, Power2Mean: 0.673038

**** COMMAND INPUT ***
**** COMMAND INPUT *** FILL TOL=1.0E-06

Tetrahedral element quality on 3 cells
Tetrahedra: 71314, Max: 1.61601, Min: 0.0922769, RMS: 0.786103, Power2Mean: 0.474601, Worst cell: 3

Total nodes:    18131
Tetrahedra:     71314
Total volume: 677903.4
**** COMMAND INPUT ***
**** COMMAND INPUT *** MODEL DELETE
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=PICK
**** COMMAND INPUT *** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='Unnamed:AAA-001' IDENTIFIER=A.00013
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** COMMAND INPUT *** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=2 BACKOFFSET=0.75
**** COMMAND INPUT ***
**** COMMAND INPUT *** MODEL CREATE
**** COMMAND INPUT ***
**** COMMAND INPUT *** MESH GENERATOR=AUTOMATIC SIZE=10.0 NORMALTOL=30.0 SURFACETOL=0.0 TOLERANCE=1.0E-06 TYPE=PREFERTETRA
Mesh generator Type II
72 faces completed, generating 36884 surface elements

Triangular facet quality on 72 faces
Triangles: 36884, Max: 0.999999, Min: 0.167905, Mean: 0.937969, Power2Mean: 0.673038

**** COMMAND INPUT ***
**** COMMAND INPUT *** FILL TOL=1.0E-06

Tetrahedral element quality on 3 cells
Tetrahedra: 71314, Max: 1.61601, Min: 0.0922769, RMS: 0.786103, Power2Mean: 0.474601, Worst cell: 3

Total nodes:    18131
Tetrahedra:     71314
Total volume: 677903.4
**** COMMAND INPUT ***
**** COMMAND INPUT *** MODEL DELETE
**** COMMAND INPUT ***
**** COMMAND INPUT *** BACKGROUND OPTION=LOAD
**** COMMAND INPUT ***
**** COMMAND INPUT *** BACKGROUND OPTION=SET SHAPE=BLOCK SCALEX=1 SCALEY=1 SCALEZ=1   XYSYMMETRYPLANE=NO YZSYMMETRYPLANE=NO ZXSYMMETRYPLANE=NO ROTZNUM=1
**** COMMAND INPUT ***
**** COMMAND INPUT *** MODEL CREATE
**** COMMAND INPUT ***
**** COMMAND INPUT *** MESH GENERATOR=AUTOMATIC SIZE=10.0 NORMALTOL=30.0 SURFACETOL=0.0 TOLERANCE=1.0E-06 TYPE=PREFERTETRA
Mesh generator Type II
79 faces completed, generating 49624 surface elements

Triangular facet quality on 79 faces
Triangles: 49624, Max: 1.0, Min: 0.151257, Mean: 0.947733, Power2Mean: 0.722283

**** COMMAND INPUT ***
**** COMMAND INPUT *** FILL TOL=1.0E-06

Tetrahedral element quality on 4 cells
Tetrahedra: 248876, Max: 1.714387, Min: 0.0319999, RMS: 0.96238, Power2Mean: 0.708068, Worst cell: 4

Total nodes:    47367
Tetrahedra:     248876
Total volume: 2.31645E+07
**** COMMAND INPUT ***
*** CONTROL COMMAND ** $ COMI 'C:\Users\mzhao\OPERA_miniMSR\layer_meshing_version.comi'
Opening file for input: C:\Users\mzhao\OPERA_miniMSR\layer_meshing_version.comi
*** COMMENT ********** // this script creates model of mini-MSR in OPERA with layered meshing
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
*** COMMENT ********** / set material to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_front'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='inner_mumetal_door' ELEMENTTYPE=Linear LEVEL=#mu_DSL SIZE#mu_flat_MES NORMALTOL=#mu_flat_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / set layering for meshing
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_front' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=2 BACKOFFSET=0.00075
**** FILE INPUT ******
*** COMMENT ********** // back
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-018'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_back' UNIQUENAME='inner_back'
*** COMMENT ********** / set material to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_back'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='inner_mumetal_door' ELEMENTTYPE=Linear LEVEL=#mu_DSL SIZE#mu_flat_MES NORMALTOL=#mu_flat_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / set layering for meshing
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_back' IDENTIFIER=A.00010
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=2 BACKOFFSET=0.00075
**** FILE INPUT ******
*** COMMENT ********** // left
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-010'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_left' UNIQUENAME='inner_left'
*** COMMENT ********** / set material to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_left'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='inner_mumetal_door' ELEMENTTYPE=Linear LEVEL=#mu_DSL SIZE#mu_flat_MES NORMALTOL=#mu_flat_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / set layering for meshing
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_left' IDENTIFIER=A.00010
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=2 BACKOFFSET=0.00075
**** FILE INPUT ******
*** COMMENT ********** // right
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-006'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_right' UNIQUENAME='inner_right'
*** COMMENT ********** / set material to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_right'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='inner_mumetal_door' ELEMENTTYPE=Linear LEVEL=#mu_DSL SIZE#mu_flat_MES NORMALTOL=#mu_flat_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / set layering for meshing
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_right' IDENTIFIER=A.00010
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=2 BACKOFFSET=0.00075
**** FILE INPUT ******
*** COMMENT ********** // top
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-017'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_top' UNIQUENAME='inner_top'
*** COMMENT ********** / set material to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_top'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='inner_mumetal_floor' ELEMENTTYPE=Linear LEVEL=#mu_DSL SIZE#mu_flat_MES NORMALTOL=#mu_flat_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / set layering for meshing
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_top' IDENTIFIER=A.00010
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=2 BACKOFFSET=0.00075
**** FILE INPUT ******
*** COMMENT ********** // bottom
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-007'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_bottom' UNIQUENAME='inner_bottom'
*** COMMENT ********** / set material to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_bottom'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='inner_mumetal_floor' ELEMENTTYPE=Linear LEVEL=#mu_DSL SIZE#mu_flat_MES NORMALTOL=#mu_flat_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / set layering for meshing
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_top' IDENTIFIER=A.00010
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=2 BACKOFFSET=0.00075
**** FILE INPUT ******
*** COMMENT ********** /// brackets
*** COMMENT ********** // front-top
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-005'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_brkt_front_top' UNIQUENAME='inner_brkt_front_top'
*** COMMENT ********** / set material to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_brkt_front_top'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='inner_mumetal_door' ELEMENTTYPE=Quadratic LEVEL=#mu_DSL SIZE#mu_bent_MES NORMALTOL=#mu_bent_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / set layering for meshing
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_front_top' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=2 BACKOFFSET=0.00075
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_front_top' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=2 BACKOFFSET=0.00075
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_front_top' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=2 BACKOFFSET=0.00075
**** FILE INPUT ******
*** COMMENT ********** // front-bottom
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-003'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_brkt_front_bottom' UNIQUENAME='inner_brkt_front_bottom'
*** COMMENT ********** / set material to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_brkt_front_bottom'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='inner_mumetal_door' ELEMENTTYPE=Quadratic LEVEL=#mu_DSL SIZE#mu_bent_MES NORMALTOL=#mu_bent_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / set layering for meshing
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_front_bottom' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=2 BACKOFFSET=0.00075
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_front_bottom' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=2 BACKOFFSET=0.00075
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_front_bottom' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=2 BACKOFFSET=0.00075
**** FILE INPUT ******
*** COMMENT ********** // front-left
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-004'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_brkt_front_left' UNIQUENAME='inner_brkt_front_left'
*** COMMENT ********** / set material to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_brkt_front_left'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='inner_mumetal_door' ELEMENTTYPE=Quadratic LEVEL=#mu_DSL SIZE#mu_bent_MES NORMALTOL=#mu_bent_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / set layering for meshing
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_front_left' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=2 BACKOFFSET=0.00075
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_front_left' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=2 BACKOFFSET=0.00075
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_front_left' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=2 BACKOFFSET=0.00075
**** FILE INPUT ******
*** COMMENT ********** // front-right
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-002'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_brkt_front_right' UNIQUENAME='inner_brkt_front_right'
*** COMMENT ********** / set material to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_brkt_front_right'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='inner_mumetal_door' ELEMENTTYPE=Quadratic LEVEL=#mu_DSL SIZE#mu_bent_MES NORMALTOL=#mu_bent_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / set layering for meshing
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_front_right' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=2 BACKOFFSET=0.00075
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_front_right' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=2 BACKOFFSET=0.00075
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_front_right' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=2 BACKOFFSET=0.00075
**** FILE INPUT ******
*** COMMENT ********** // bottom-left
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-012'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_brkt_bottom_left' UNIQUENAME='inner_brkt_bottom_left'
*** COMMENT ********** / set material to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_brkt_bottom_left'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='inner_mumetal_floor' ELEMENTTYPE=Quadratic LEVEL=#mu_DSL SIZE#mu_bent_MES NORMALTOL=#mu_bent_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / set layering for meshing
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_bottom_left' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=2 BACKOFFSET=0.00075
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_bottom_left' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=2 BACKOFFSET=0.00075
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_bottom_left' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=2 BACKOFFSET=0.00075
**** FILE INPUT ******
*** COMMENT ********** // bottom-right
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-008'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_brkt_bottom_right' UNIQUENAME='inner_brkt_bottom_right'
*** COMMENT ********** / set material to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_brkt_bottom_right'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='inner_mumetal_floor' ELEMENTTYPE=Quadratic LEVEL=#mu_DSL SIZE#mu_bent_MES NORMALTOL=#mu_bent_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / set layering for meshing
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_bottom_right' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=2 BACKOFFSET=0.00075
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_bottom_right' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=2 BACKOFFSET=0.00075
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_bottom_right' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=2 BACKOFFSET=0.00075
**** FILE INPUT ******
*** COMMENT ********** // top-left
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-011'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_brkt_top_left' UNIQUENAME='inner_brkt_top_left'
*** COMMENT ********** / set material to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_brkt_top_left'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='inner_mumetal_floor' ELEMENTTYPE=Quadratic LEVEL=#mu_DSL SIZE#mu_bent_MES NORMALTOL=#mu_bent_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / set layering for meshing
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_top_left' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=2 BACKOFFSET=0.00075
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_top_left' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=2 BACKOFFSET=0.00075
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_top_left' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=2 BACKOFFSET=0.00075
**** FILE INPUT ******
*** COMMENT ********** // top-right
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-009'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_brkt_top_right' UNIQUENAME='inner_brkt_top_right'
*** COMMENT ********** / set material to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_brkt_top_right'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='inner_mumetal_floor' ELEMENTTYPE=Quadratic LEVEL=#mu_DSL SIZE#mu_bent_MES NORMALTOL=#mu_bent_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / set layering for meshing
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_top_right' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=2 BACKOFFSET=0.00075
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_top_right' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=2 BACKOFFSET=0.00075
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_top_right' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=2 BACKOFFSET=0.00075
**** FILE INPUT ******
*** COMMENT ********** // back-bottom
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-013'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_brkt_back_bottom' UNIQUENAME='inner_brkt_back_bottom'
*** COMMENT ********** / set material to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_brkt_back_bottom'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='inner_mumetal_door' ELEMENTTYPE=Quadratic LEVEL=#mu_DSL SIZE#mu_bent_MES NORMALTOL=#mu_bent_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / set layering for meshing
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_back_bottom' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=2 BACKOFFSET=0.00075
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_back_bottom' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=2 BACKOFFSET=0.00075
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_back_bottom' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=2 BACKOFFSET=0.00075
**** FILE INPUT ******
*** COMMENT ********** // back-top
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-015'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_brkt_back_top' UNIQUENAME='inner_brkt_back_top'
*** COMMENT ********** / set material to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_brkt_back_top'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='inner_mumetal_door' ELEMENTTYPE=Quadratic LEVEL=#mu_DSL SIZE#mu_bent_MES NORMALTOL=#mu_bent_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / set layering for meshing
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_back_top' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=2 BACKOFFSET=0.00075
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_back_top' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=2 BACKOFFSET=0.00075
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_back_top' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=2 BACKOFFSET=0.00075
**** FILE INPUT ******
*** COMMENT ********** // back-left
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-016'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_brkt_back_left' UNIQUENAME='inner_brkt_back_left'
*** COMMENT ********** / set material to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_brkt_back_left'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='inner_mumetal_door' ELEMENTTYPE=Quadratic LEVEL=#mu_DSL SIZE#mu_bent_MES NORMALTOL=#mu_bent_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / set layering for meshing
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_back_left' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=2 BACKOFFSET=0.00075
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_back_left' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=2 BACKOFFSET=0.00075
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_back_left' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=2 BACKOFFSET=0.00075
**** FILE INPUT ******
*** COMMENT ********** // back-right
**** FILE INPUT ****** FILTER TYPE=BODY
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAA-014'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** RENAME NAME='inner_brkt_back_right' UNIQUENAME='inner_brkt_back_right'
*** COMMENT ********** / set material to mumetal
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD PROPERTY=UniqueName LABEL='inner_brkt_back_right'
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** CELLDATA OPTION=MODIFY MATERIALLABEL='inner_mumetal_door' ELEMENTTYPE=Quadratic LEVEL=#mu_DSL SIZE#mu_bent_MES NORMALTOL=#mu_bent_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** / set layering for meshing
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_back_right' IDENTIFIER=A.00018
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=2 BACKOFFSET=0.00075
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_back_right' IDENTIFIER=A.00013
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=2 BACKOFFSET=0.00075
**** FILE INPUT ****** FILTER TYPE=FACE
**** FILE INPUT ****** PREVIEW OPTION=PICK
**** FILE INPUT ****** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_back_right' IDENTIFIER=A.00006
**** FILE INPUT ****** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** FILE INPUT ****** FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=2 BACKOFFSET=0.00075
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
*** COMMENT ********** /++ / load structural step file
*** COMMENT ********** /++ LOAD OPTION=INSERT FILE='stpfiles//outer_layer.STEP'
*** COMMENT ********** /++
*** COMMENT ********** /++ /// Flat panels
*** COMMENT ********** /++ // front
*** COMMENT ********** /++ / rename body
*** COMMENT ********** /++ FILTER TYPE=BODY
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-001'
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / RENAME NAME='outer_front' UNIQUENAME='outer_front'
*** COMMENT ********** /++ / set material to mumetal
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_front'
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / CELLDATA OPTION=MODIFY MATERIALLABEL='outer_mumetal_door' ELEMENTTYPE=Linear LEVEL=#mu_DSL SIZE#mu_flat_MES NORMALTOL=#mu_flat_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** /++ / set layering for meshing
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_front' IDENTIFIER=A.00006
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0004
*** COMMENT ********** /++
*** COMMENT ********** /++ // back
*** COMMENT ********** /++ FILTER TYPE=BODY
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-026'
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / RENAME NAME='outer_back' UNIQUENAME='outer_back'
*** COMMENT ********** /++ / set material to mumetal
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_back'
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / CELLDATA OPTION=MODIFY MATERIALLABEL='outer_mumetal_door' ELEMENTTYPE=Linear LEVEL=#mu_DSL SIZE#mu_flat_MES NORMALTOL=#mu_flat_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** /++ / set layering for meshing
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_back' IDENTIFIER=A.00010
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0004
*** COMMENT ********** /++
*** COMMENT ********** /++ // left
*** COMMENT ********** /++ FILTER TYPE=BODY
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-014'
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / RENAME NAME='outer_left' UNIQUENAME='outer_left'
*** COMMENT ********** /++ / set material to mumetal
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_left'
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / CELLDATA OPTION=MODIFY MATERIALLABEL='outer_mumetal_door' ELEMENTTYPE=Linear LEVEL=#mu_DSL SIZE#mu_flat_MES NORMALTOL=#mu_flat_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** /++ / set layering for meshing
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_left' IDENTIFIER=A.00010
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0004
*** COMMENT ********** /++
*** COMMENT ********** /++ // right
*** COMMENT ********** /++ FILTER TYPE=BODY
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-018'
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / RENAME NAME='outer_right' UNIQUENAME='outer_right'
*** COMMENT ********** /++ / set material to mumetal
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_right'
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / CELLDATA OPTION=MODIFY MATERIALLABEL='outer_mumetal_door' ELEMENTTYPE=Linear LEVEL=#mu_DSL SIZE#mu_flat_MES NORMALTOL=#mu_flat_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** /++ / set layering for meshing
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_right' IDENTIFIER=A.00010
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0004
*** COMMENT ********** /++
*** COMMENT ********** /++ // top
*** COMMENT ********** /++ FILTER TYPE=BODY
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-025'
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / RENAME NAME='outer_top' UNIQUENAME='outer_top'
*** COMMENT ********** /++ / set material to mumetal
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_top'
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / CELLDATA OPTION=MODIFY MATERIALLABEL='outer_mumetal_floor' ELEMENTTYPE=Linear LEVEL=#mu_DSL SIZE#mu_flat_MES NORMALTOL=#mu_flat_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** /++ / set layering for meshing
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_top' IDENTIFIER=A.00010
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0004
*** COMMENT ********** /++
*** COMMENT ********** /++ // bottom
*** COMMENT ********** /++ FILTER TYPE=BODY
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-005'
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / RENAME NAME='outer_bottom' UNIQUENAME='outer_bottom'
*** COMMENT ********** /++ / set material to mumetal
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_bottom'
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / CELLDATA OPTION=MODIFY MATERIALLABEL='outer_mumetal_floor' ELEMENTTYPE=Linear LEVEL=#mu_DSL SIZE#mu_flat_MES NORMALTOL=#mu_flat_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** /++ / set layering for meshing
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_top' IDENTIFIER=A.00010
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0004
*** COMMENT ********** /++
*** COMMENT ********** /++ /// brackets
*** COMMENT ********** /++ // front-top
*** COMMENT ********** /++ FILTER TYPE=BODY
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-021'
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / RENAME NAME='outer_brkt_front_top' UNIQUENAME='outer_brkt_front_top'
*** COMMENT ********** /++ / set material to mumetal
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_brkt_front_top'
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / CELLDATA OPTION=MODIFY MATERIALLABEL='outer_mumetal_door' ELEMENTTYPE=Quadratic LEVEL=#mu_DSL SIZE#mu_bent_MES NORMALTOL=#mu_bent_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** /++ / set layering for meshing
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_front_top' IDENTIFIER=A.00018
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0004
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_front_top' IDENTIFIER=A.00013
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0004
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_front_top' IDENTIFIER=A.00006
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0004
*** COMMENT ********** /++
*** COMMENT ********** /++ // front-bottom
*** COMMENT ********** /++ FILTER TYPE=BODY
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-003'
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / RENAME NAME='outer_brkt_front_bottom' UNIQUENAME='outer_brkt_front_bottom'
*** COMMENT ********** /++ / set material to mumetal
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_brkt_front_bottom'
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / CELLDATA OPTION=MODIFY MATERIALLABEL='outer_mumetal_door' ELEMENTTYPE=Quadratic LEVEL=#mu_DSL SIZE#mu_bent_MES NORMALTOL=#mu_bent_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** /++ / set layering for meshing
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_front_bottom' IDENTIFIER=A.00018
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0004
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_front_bottom' IDENTIFIER=A.00013
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0004
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_front_bottom' IDENTIFIER=A.00006
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0004
*** COMMENT ********** /++
*** COMMENT ********** /++ // front-left
*** COMMENT ********** /++ FILTER TYPE=BODY
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-013'
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / RENAME NAME='outer_brkt_front_left' UNIQUENAME='outer_brkt_front_left'
*** COMMENT ********** /++ / set material to mumetal
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_brkt_front_left'
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / CELLDATA OPTION=MODIFY MATERIALLABEL='outer_mumetal_door' ELEMENTTYPE=Quadratic LEVEL=#mu_DSL SIZE#mu_bent_MES NORMALTOL=#mu_bent_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** /++ / set layering for meshing
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_front_left' IDENTIFIER=A.00018
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0004
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_front_left' IDENTIFIER=A.00013
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0004
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_front_left' IDENTIFIER=A.00006
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0004
*** COMMENT ********** /++
*** COMMENT ********** /++ // front-right
*** COMMENT ********** /++ FILTER TYPE=BODY
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-017'
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / RENAME NAME='outer_brkt_front_right' UNIQUENAME='outer_brkt_front_right'
*** COMMENT ********** /++ / set material to mumetal
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_brkt_front_right'
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / CELLDATA OPTION=MODIFY MATERIALLABEL='outer_mumetal_door' ELEMENTTYPE=Quadratic LEVEL=#mu_DSL SIZE#mu_bent_MES NORMALTOL=#mu_bent_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** /++ / set layering for meshing
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_front_right' IDENTIFIER=A.00018
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0004
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_front_right' IDENTIFIER=A.00013
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0004
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_front_right' IDENTIFIER=A.00006
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0004
*** COMMENT ********** /++
*** COMMENT ********** /++ // bottom-left
*** COMMENT ********** /++ FILTER TYPE=BODY
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-015'
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / RENAME NAME='outer_brkt_bottom_left' UNIQUENAME='outer_brkt_bottom_left'
*** COMMENT ********** /++ / set material to mumetal
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_brkt_bottom_left'
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / CELLDATA OPTION=MODIFY MATERIALLABEL='outer_mumetal_floor' ELEMENTTYPE=Quadratic LEVEL=#mu_DSL SIZE#mu_bent_MES NORMALTOL=#mu_bent_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** /++ / set layering for meshing
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_bottom_left' IDENTIFIER=A.00018
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0004
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_bottom_left' IDENTIFIER=A.00013
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0004
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_bottom_left' IDENTIFIER=A.00006
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic
*** COMMENT ********** /++
*** COMMENT ********** /++ // bottom-right
*** COMMENT ********** /++ FILTER TYPE=BODY
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-019'
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / RENAME NAME='outer_brkt_bottom_right' UNIQUENAME='outer_brkt_bottom_right'
*** COMMENT ********** /++ / set material to mumetal
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_brkt_bottom_right'
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / CELLDATA OPTION=MODIFY MATERIALLABEL='outer_mumetal_floor' ELEMENTTYPE=Quadratic LEVEL=#mu_DSL SIZE#mu_bent_MES NORMALTOL=#mu_bent_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** /++ / set layering for meshing
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_bottom_right' IDENTIFIER=A.00018
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0004
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_bottom_right' IDENTIFIER=A.00013
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0004
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_bottom_right' IDENTIFIER=A.00006
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic
*** COMMENT ********** /++
*** COMMENT ********** /++ // top-left
*** COMMENT ********** /++ FILTER TYPE=BODY
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-022'
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / RENAME NAME='outer_brkt_top_left' UNIQUENAME='outer_brkt_top_left'
*** COMMENT ********** /++ / set material to mumetal
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_brkt_top_left'
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / CELLDATA OPTION=MODIFY MATERIALLABEL='outer_mumetal_floor' ELEMENTTYPE=Quadratic LEVEL=#mu_DSL SIZE#mu_bent_MES NORMALTOL=#mu_bent_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** /++ / set layering for meshing
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_top_left' IDENTIFIER=A.00018
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0004
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_top_left' IDENTIFIER=A.00013
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0004
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_top_left' IDENTIFIER=A.00006
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic
*** COMMENT ********** /++
*** COMMENT ********** /++ // top-right
*** COMMENT ********** /++ FILTER TYPE=BODY
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-024'
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / RENAME NAME='outer_brkt_top_right' UNIQUENAME='outer_brkt_top_right'
*** COMMENT ********** /++ / set material to mumetal
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_brkt_top_right'
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / CELLDATA OPTION=MODIFY MATERIALLABEL='outer_mumetal_floor' ELEMENTTYPE=Quadratic LEVEL=#mu_DSL SIZE#mu_bent_MES NORMALTOL=#mu_bent_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** /++ / set layering for meshing
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_top_right' IDENTIFIER=A.00018
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0004
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_top_right' IDENTIFIER=A.00013
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0004
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_top_right' IDENTIFIER=A.00006
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic
*** COMMENT ********** /++
*** COMMENT ********** /++ // back-bottom
*** COMMENT ********** /++ FILTER TYPE=BODY
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-020'
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / RENAME NAME='outer_brkt_back_bottom' UNIQUENAME='outer_brkt_back_bottom'
*** COMMENT ********** /++ / set material to mumetal
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_brkt_back_bottom'
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / CELLDATA OPTION=MODIFY MATERIALLABEL='outer_mumetal_door' ELEMENTTYPE=Quadratic LEVEL=#mu_DSL SIZE#mu_bent_MES NORMALTOL=#mu_bent_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** /++ / set layering for meshing
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_back_bottom' IDENTIFIER=A.00018
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0004
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_back_bottom' IDENTIFIER=A.00013
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0004
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_back_bottom' IDENTIFIER=A.00006
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic
*** COMMENT ********** /++
*** COMMENT ********** /++ // back-top
*** COMMENT ********** /++ FILTER TYPE=BODY
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-023'
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / RENAME NAME='outer_brkt_back_top' UNIQUENAME='outer_brkt_back_top'
*** COMMENT ********** /++ / set material to mumetal
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_brkt_back_top'
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / CELLDATA OPTION=MODIFY MATERIALLABEL='outer_mumetal_door' ELEMENTTYPE=Quadratic LEVEL=#mu_DSL SIZE#mu_bent_MES NORMALTOL=#mu_bent_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** /++ / set layering for meshing
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_back_top' IDENTIFIER=A.00018
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0004
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_back_top' IDENTIFIER=A.00013
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0004
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_back_top' IDENTIFIER=A.00006
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic
*** COMMENT ********** /++
*** COMMENT ********** /++ // back-left
*** COMMENT ********** /++ FILTER TYPE=BODY
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-012'
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / RENAME NAME='outer_brkt_back_left' UNIQUENAME='outer_brkt_back_left'
*** COMMENT ********** /++ / set material to mumetal
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_brkt_back_left'
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / CELLDATA OPTION=MODIFY MATERIALLABEL='outer_mumetal_door' ELEMENTTYPE=Quadratic LEVEL=#mu_DSL SIZE#mu_bent_MES NORMALTOL=#mu_bent_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** /++ / set layering for meshing
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_back_left' IDENTIFIER=A.00018
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0004
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_back_left' IDENTIFIER=A.00013
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0004
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_back_left' IDENTIFIER=A.00006
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic
*** COMMENT ********** /++
*** COMMENT ********** /++ // back-right
*** COMMENT ********** /++ FILTER TYPE=BODY
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD PROPERTY=UniqueName LABEL='Unnamed:AAB-016'
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / RENAME NAME='outer_brkt_back_right' UNIQUENAME='outer_brkt_back_right'
*** COMMENT ********** /++ / set material to mumetal
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD PROPERTY=UniqueName LABEL='outer_brkt_back_right'
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / CELLDATA OPTION=MODIFY MATERIALLABEL='outer_mumetal_door' ELEMENTTYPE=Quadratic LEVEL=#mu_DSL SIZE#mu_bent_MES NORMALTOL=#mu_bent_MAbE ELEMSHAPEPREF=NONE
*** COMMENT ********** /++ / set layering for meshing
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_back_right' IDENTIFIER=A.00018
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0004
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_back_right' IDENTIFIER=A.00013
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=5 BACKOFFSET=0.0004
*** COMMENT ********** /++ FILTER TYPE=FACE
*** COMMENT ********** /++ PREVIEW OPTION=PICK
*** COMMENT ********** / PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='outer_brkt_back_right' IDENTIFIER=A.00006
*** COMMENT ********** /++ PREVIEW OPTION=ACCEPT REDISPLAY=NO
*** COMMENT ********** / FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic
*** COMMENT ********** /++
*** COMMENT ********** /++ // BH data
*** COMMENT ********** /++ / create labels
*** COMMENT ********** /++ BHDATA OPTION=NEW  LABEL='outermumetal_door_BH'
*** COMMENT ********** /++ BHDATA OPTION=NEW  LABEL='outermumetal_floor_BH'
*** COMMENT ********** /++
*** COMMENT ********** /++ / assign bh file to label
*** COMMENT ********** /++ BHDATA OPTION=LOAD LABEL=outermumetal_door_BH FILE='bhdata//miniMSR_outer_door_anh.bh'
*** COMMENT ********** /++ BHDATA OPTION=LOAD LABEL=outermumetal_floor_BH FILE='bhdata//miniMSR_outer_floor_anh.bh'
*** COMMENT ********** /++
*** COMMENT ********** /++ / match material label to bh label
*** COMMENT ********** /++ MATERIALS UNPICK
*** COMMENT ********** / MATERIALS GUIINIT
*** COMMENT ********** /++ MATERIALS PICK 'outer_mumetal_door'
*** COMMENT ********** /++ MATERIALS OPTION=MODIFY MULINEARITY=NONLINEAR MUANISOTROPY=ISOTROPIC BH='outermumetal_door_BH'
*** COMMENT ********** / MATERIALS UNPICK
*** COMMENT ********** /++ MATERIALS UNPICK
*** COMMENT ********** / MATERIALS GUIINIT
*** COMMENT ********** /++ MATERIALS PICK 'outer_mumetal_floor'
*** COMMENT ********** /++ MATERIALS OPTION=MODIFY MULINEARITY=NONLINEAR MUANISOTROPY=ISOTROPIC BH='outermumetal_floor_BH'
*** COMMENT ********** / MATERIALS UNPICK
**** FILE INPUT ******
*** COMMENT ********** //// Symmetry
**** FILE INPUT ****** BACKGROUND OPTION=LOAD
**** FILE INPUT ****** BACKGROUND OPTION=SET SHAPE=BLOCK SCALEX=5 SCALEY=5 SCALEZ=5   XYSYMMETRYPLANE=NO YZSYMMETRYPLANE=YES ROTZNUM=1 ZXSYMMETRYPLANE=YES EMRYZ=TANGMAGN EMRZX=NORMMAGN
**** FILE INPUT ******
*** COMMENT ********** /++ //// Create Model Body
*** COMMENT ********** /++ MODEL CREATE
*** COMMENT ********** /++
*** COMMENT ********** /++ //// Meshing
*** COMMENT ********** /++ / Surface Mesh
*** COMMENT ********** /++ MESH GENERATOR=AUTOMATIC SIZE=0.1 NORMALTOL=30.0 SURFACETOL=0.0 TOLERANCE=1.0E-06 TYPE=PREFERTETRA
*** COMMENT ********** /++
*** COMMENT ********** /++ / Volume Mesh
*** COMMENT ********** /++ FILL TOL=1.0E-05
**** FILE INPUT ******
*** COMMENT ********** //// Create Database
*** COMMENT ********** /++ SOLVERS SOLVENOW=NO SAVEMODEL=YES,
*** COMMENT ********** / SOLVERS OPTION=TEST FILE='mini_MSR' UNITS=METRE ELEMENT=MIXED SURFACE=CURVED
*** COMMENT ********** / COMMENT CLEAR=YES TYPE=DBTITLE
*** COMMENT ********** / SOLVERS OPTION=OVERWRITE
**** FILE INPUT ******
**** FILE INPUT ******
**** FILE INPUT ******
**** FILE INPUT ******
**** FILE INPUT ******
**** FILE INPUT ******
**** FILE INPUT ******
**** FILE INPUT ******
**** FILE INPUT ******
**** FILE INPUT ******
**** FILE INPUT ******
**** FILE INPUT ******
**** FILE INPUT ******
**** FILE INPUT ******
**** FILE INPUT ******
**** FILE INPUT ******
**** FILE INPUT ******
**** FILE INPUT ******
**** FILE INPUT ******
**** FILE INPUT ******
**** FILE INPUT ******
**** FILE INPUT ******
**** FILE INPUT ******
**** FILE INPUT ******
**** FILE INPUT ******
**** FILE INPUT ******
**** FILE INPUT ******
**** FILE INPUT ******
**** FILE INPUT ******
**** FILE INPUT ******
End of command file.
*** CONTROL COMMAND ** $ END COMI
**** COMMAND INPUT ***
**** COMMAND INPUT *** FILTER COMMAND=CURSORHIDE
**** COMMAND INPUT ***
**** COMMAND INPUT *** SELECT OPTION=CURSORHIDE TYPE=FACE UNIQUEBODYNAME='inner_back' IDENTIFIER=A.00009
**** COMMAND INPUT ***
**** COMMAND INPUT *** SELECT OPTION=CURSORHIDE TYPE=FACE UNIQUEBODYNAME='inner_back' IDENTIFIER=A.00010
**** COMMAND INPUT ***
**** COMMAND INPUT *** FILTER COMMAND=PICK
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=PICK
**** COMMAND INPUT *** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_front_right' IDENTIFIER=A.00018
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** COMMAND INPUT *** FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=2 BACKOFFSET=0.00075
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=PICK
**** COMMAND INPUT *** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_front_right' IDENTIFIER=A.00013
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** COMMAND INPUT *** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=2 BACKOFFSET=0.00075
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=PICK
**** COMMAND INPUT *** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_front_right' IDENTIFIER=A.00006
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** COMMAND INPUT *** FACEDATA OPTION=MODIFY ELEMENTTYPE=Linear FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=2 BACKOFFSET=0.00075
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=PICK
**** COMMAND INPUT *** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_brkt_front_right' IDENTIFIER=A.00013
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=ACCEPT REDISPLAY=NO
**** COMMAND INPUT *** FACEDATA OPTION=MODIFY ELEMENTTYPE=Quadratic FORMETHOD=NONE BACKMETHOD=GEOMETRY BACKLAYERS=2 BACKOFFSET=0.00075
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=PICK
**** COMMAND INPUT *** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_front' IDENTIFIER=A.00006
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT *** SELECT OPTION=DEFAULT
**** COMMAND INPUT ***
**** COMMAND INPUT *** PICK OPTION=TOGGLE TYPE=FACE UNIQUEBODYNAME='inner_back' IDENTIFIER=A.00009
**** COMMAND INPUT ***
**** COMMAND INPUT *** PICK OPTION=TOGGLE TYPE=FACE UNIQUEBODYNAME='inner_back' IDENTIFIER=A.00009
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=PICK
**** COMMAND INPUT ***
**** COMMAND INPUT *** PICK OPTION=ADD TYPE=FACE UNIQUEBODYNAME='inner_back' IDENTIFIER=A.00009
**** COMMAND INPUT ***
**** COMMAND INPUT ***
**** COMMAND INPUT *** PREVIEW OPTION=UNPICK REDISPLAY=YES
**** COMMAND INPUT ***
