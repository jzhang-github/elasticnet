��
��
B
AssignVariableOp
resource
value"dtype"
dtypetype�
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
;
Elu
features"T
activations"T"
Ttype:
2
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(�

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring �
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.4.02v2.4.0-rc4-71-g582c8d236cb8�

�
Hidden_layer_0/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:P*&
shared_nameHidden_layer_0/kernel

)Hidden_layer_0/kernel/Read/ReadVariableOpReadVariableOpHidden_layer_0/kernel*
_output_shapes

:P*
dtype0
~
Hidden_layer_0/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*$
shared_nameHidden_layer_0/bias
w
'Hidden_layer_0/bias/Read/ReadVariableOpReadVariableOpHidden_layer_0/bias*
_output_shapes
:P*
dtype0
�
Hidden_layer_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:PP*&
shared_nameHidden_layer_1/kernel

)Hidden_layer_1/kernel/Read/ReadVariableOpReadVariableOpHidden_layer_1/kernel*
_output_shapes

:PP*
dtype0
~
Hidden_layer_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*$
shared_nameHidden_layer_1/bias
w
'Hidden_layer_1/bias/Read/ReadVariableOpReadVariableOpHidden_layer_1/bias*
_output_shapes
:P*
dtype0
�
Hidden_layer_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:PP*&
shared_nameHidden_layer_2/kernel

)Hidden_layer_2/kernel/Read/ReadVariableOpReadVariableOpHidden_layer_2/kernel*
_output_shapes

:PP*
dtype0
~
Hidden_layer_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*$
shared_nameHidden_layer_2/bias
w
'Hidden_layer_2/bias/Read/ReadVariableOpReadVariableOpHidden_layer_2/bias*
_output_shapes
:P*
dtype0
�
Hidden_layer_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:PP*&
shared_nameHidden_layer_3/kernel

)Hidden_layer_3/kernel/Read/ReadVariableOpReadVariableOpHidden_layer_3/kernel*
_output_shapes

:PP*
dtype0
~
Hidden_layer_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*$
shared_nameHidden_layer_3/bias
w
'Hidden_layer_3/bias/Read/ReadVariableOpReadVariableOpHidden_layer_3/bias*
_output_shapes
:P*
dtype0
�
Output_layer/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:P*$
shared_nameOutput_layer/kernel
{
'Output_layer/kernel/Read/ReadVariableOpReadVariableOpOutput_layer/kernel*
_output_shapes

:P*
dtype0
z
Output_layer/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameOutput_layer/bias
s
%Output_layer/bias/Read/ReadVariableOpReadVariableOpOutput_layer/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_2
[
total_2/Read/ReadVariableOpReadVariableOptotal_2*
_output_shapes
: *
dtype0
b
count_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_2
[
count_2/Read/ReadVariableOpReadVariableOpcount_2*
_output_shapes
: *
dtype0
�
Adam/Hidden_layer_0/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:P*-
shared_nameAdam/Hidden_layer_0/kernel/m
�
0Adam/Hidden_layer_0/kernel/m/Read/ReadVariableOpReadVariableOpAdam/Hidden_layer_0/kernel/m*
_output_shapes

:P*
dtype0
�
Adam/Hidden_layer_0/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*+
shared_nameAdam/Hidden_layer_0/bias/m
�
.Adam/Hidden_layer_0/bias/m/Read/ReadVariableOpReadVariableOpAdam/Hidden_layer_0/bias/m*
_output_shapes
:P*
dtype0
�
Adam/Hidden_layer_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:PP*-
shared_nameAdam/Hidden_layer_1/kernel/m
�
0Adam/Hidden_layer_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/Hidden_layer_1/kernel/m*
_output_shapes

:PP*
dtype0
�
Adam/Hidden_layer_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*+
shared_nameAdam/Hidden_layer_1/bias/m
�
.Adam/Hidden_layer_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/Hidden_layer_1/bias/m*
_output_shapes
:P*
dtype0
�
Adam/Hidden_layer_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:PP*-
shared_nameAdam/Hidden_layer_2/kernel/m
�
0Adam/Hidden_layer_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/Hidden_layer_2/kernel/m*
_output_shapes

:PP*
dtype0
�
Adam/Hidden_layer_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*+
shared_nameAdam/Hidden_layer_2/bias/m
�
.Adam/Hidden_layer_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/Hidden_layer_2/bias/m*
_output_shapes
:P*
dtype0
�
Adam/Hidden_layer_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:PP*-
shared_nameAdam/Hidden_layer_3/kernel/m
�
0Adam/Hidden_layer_3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/Hidden_layer_3/kernel/m*
_output_shapes

:PP*
dtype0
�
Adam/Hidden_layer_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*+
shared_nameAdam/Hidden_layer_3/bias/m
�
.Adam/Hidden_layer_3/bias/m/Read/ReadVariableOpReadVariableOpAdam/Hidden_layer_3/bias/m*
_output_shapes
:P*
dtype0
�
Adam/Output_layer/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:P*+
shared_nameAdam/Output_layer/kernel/m
�
.Adam/Output_layer/kernel/m/Read/ReadVariableOpReadVariableOpAdam/Output_layer/kernel/m*
_output_shapes

:P*
dtype0
�
Adam/Output_layer/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/Output_layer/bias/m
�
,Adam/Output_layer/bias/m/Read/ReadVariableOpReadVariableOpAdam/Output_layer/bias/m*
_output_shapes
:*
dtype0
�
Adam/Hidden_layer_0/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:P*-
shared_nameAdam/Hidden_layer_0/kernel/v
�
0Adam/Hidden_layer_0/kernel/v/Read/ReadVariableOpReadVariableOpAdam/Hidden_layer_0/kernel/v*
_output_shapes

:P*
dtype0
�
Adam/Hidden_layer_0/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*+
shared_nameAdam/Hidden_layer_0/bias/v
�
.Adam/Hidden_layer_0/bias/v/Read/ReadVariableOpReadVariableOpAdam/Hidden_layer_0/bias/v*
_output_shapes
:P*
dtype0
�
Adam/Hidden_layer_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:PP*-
shared_nameAdam/Hidden_layer_1/kernel/v
�
0Adam/Hidden_layer_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/Hidden_layer_1/kernel/v*
_output_shapes

:PP*
dtype0
�
Adam/Hidden_layer_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*+
shared_nameAdam/Hidden_layer_1/bias/v
�
.Adam/Hidden_layer_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/Hidden_layer_1/bias/v*
_output_shapes
:P*
dtype0
�
Adam/Hidden_layer_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:PP*-
shared_nameAdam/Hidden_layer_2/kernel/v
�
0Adam/Hidden_layer_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/Hidden_layer_2/kernel/v*
_output_shapes

:PP*
dtype0
�
Adam/Hidden_layer_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*+
shared_nameAdam/Hidden_layer_2/bias/v
�
.Adam/Hidden_layer_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/Hidden_layer_2/bias/v*
_output_shapes
:P*
dtype0
�
Adam/Hidden_layer_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:PP*-
shared_nameAdam/Hidden_layer_3/kernel/v
�
0Adam/Hidden_layer_3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/Hidden_layer_3/kernel/v*
_output_shapes

:PP*
dtype0
�
Adam/Hidden_layer_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*+
shared_nameAdam/Hidden_layer_3/bias/v
�
.Adam/Hidden_layer_3/bias/v/Read/ReadVariableOpReadVariableOpAdam/Hidden_layer_3/bias/v*
_output_shapes
:P*
dtype0
�
Adam/Output_layer/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:P*+
shared_nameAdam/Output_layer/kernel/v
�
.Adam/Output_layer/kernel/v/Read/ReadVariableOpReadVariableOpAdam/Output_layer/kernel/v*
_output_shapes

:P*
dtype0
�
Adam/Output_layer/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/Output_layer/bias/v
�
,Adam/Output_layer/bias/v/Read/ReadVariableOpReadVariableOpAdam/Output_layer/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
�:
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�9
value�9B�9 B�9
�
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
layer_with_weights-4
layer-4
	optimizer
regularization_losses
	variables
	trainable_variables

	keras_api

signatures
h

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
h

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
h

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
h

kernel
bias
 regularization_losses
!	variables
"trainable_variables
#	keras_api
h

$kernel
%bias
&regularization_losses
'	variables
(trainable_variables
)	keras_api
�
*iter

+beta_1

,beta_2
	-decay
.learning_ratem^m_m`mambmcmdme$mf%mgvhvivjvkvlvmvnvo$vp%vq
 
F
0
1
2
3
4
5
6
7
$8
%9
F
0
1
2
3
4
5
6
7
$8
%9
�
/metrics
regularization_losses
	variables
0non_trainable_variables

1layers
	trainable_variables
2layer_metrics
3layer_regularization_losses
 
a_
VARIABLE_VALUEHidden_layer_0/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUEHidden_layer_0/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
�
4metrics
regularization_losses
	variables
5non_trainable_variables

6layers
trainable_variables
7layer_metrics
8layer_regularization_losses
a_
VARIABLE_VALUEHidden_layer_1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUEHidden_layer_1/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
�
9metrics
regularization_losses
	variables
:non_trainable_variables

;layers
trainable_variables
<layer_metrics
=layer_regularization_losses
a_
VARIABLE_VALUEHidden_layer_2/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUEHidden_layer_2/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
�
>metrics
regularization_losses
	variables
?non_trainable_variables

@layers
trainable_variables
Alayer_metrics
Blayer_regularization_losses
a_
VARIABLE_VALUEHidden_layer_3/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUEHidden_layer_3/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
�
Cmetrics
 regularization_losses
!	variables
Dnon_trainable_variables

Elayers
"trainable_variables
Flayer_metrics
Glayer_regularization_losses
_]
VARIABLE_VALUEOutput_layer/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEOutput_layer/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE
 

$0
%1

$0
%1
�
Hmetrics
&regularization_losses
'	variables
Inon_trainable_variables

Jlayers
(trainable_variables
Klayer_metrics
Llayer_regularization_losses
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE

M0
N1
O2
 
#
0
1
2
3
4
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
4
	Ptotal
	Qcount
R	variables
S	keras_api
D
	Ttotal
	Ucount
V
_fn_kwargs
W	variables
X	keras_api
D
	Ytotal
	Zcount
[
_fn_kwargs
\	variables
]	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

P0
Q1

R	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

T0
U1

W	variables
QO
VARIABLE_VALUEtotal_24keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_24keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUE
 

Y0
Z1

\	variables
��
VARIABLE_VALUEAdam/Hidden_layer_0/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
�~
VARIABLE_VALUEAdam/Hidden_layer_0/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEAdam/Hidden_layer_1/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
�~
VARIABLE_VALUEAdam/Hidden_layer_1/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEAdam/Hidden_layer_2/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
�~
VARIABLE_VALUEAdam/Hidden_layer_2/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEAdam/Hidden_layer_3/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
�~
VARIABLE_VALUEAdam/Hidden_layer_3/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEAdam/Output_layer/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/Output_layer/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEAdam/Hidden_layer_0/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
�~
VARIABLE_VALUEAdam/Hidden_layer_0/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEAdam/Hidden_layer_1/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
�~
VARIABLE_VALUEAdam/Hidden_layer_1/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEAdam/Hidden_layer_2/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
�~
VARIABLE_VALUEAdam/Hidden_layer_2/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEAdam/Hidden_layer_3/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
�~
VARIABLE_VALUEAdam/Hidden_layer_3/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEAdam/Output_layer/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/Output_layer/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~
serving_default_Input_layerPlaceholder*'
_output_shapes
:���������*
dtype0*
shape:���������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_Input_layerHidden_layer_0/kernelHidden_layer_0/biasHidden_layer_1/kernelHidden_layer_1/biasHidden_layer_2/kernelHidden_layer_2/biasHidden_layer_3/kernelHidden_layer_3/biasOutput_layer/kernelOutput_layer/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8� *-
f(R&
$__inference_signature_wrapper_243751
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename)Hidden_layer_0/kernel/Read/ReadVariableOp'Hidden_layer_0/bias/Read/ReadVariableOp)Hidden_layer_1/kernel/Read/ReadVariableOp'Hidden_layer_1/bias/Read/ReadVariableOp)Hidden_layer_2/kernel/Read/ReadVariableOp'Hidden_layer_2/bias/Read/ReadVariableOp)Hidden_layer_3/kernel/Read/ReadVariableOp'Hidden_layer_3/bias/Read/ReadVariableOp'Output_layer/kernel/Read/ReadVariableOp%Output_layer/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal_2/Read/ReadVariableOpcount_2/Read/ReadVariableOp0Adam/Hidden_layer_0/kernel/m/Read/ReadVariableOp.Adam/Hidden_layer_0/bias/m/Read/ReadVariableOp0Adam/Hidden_layer_1/kernel/m/Read/ReadVariableOp.Adam/Hidden_layer_1/bias/m/Read/ReadVariableOp0Adam/Hidden_layer_2/kernel/m/Read/ReadVariableOp.Adam/Hidden_layer_2/bias/m/Read/ReadVariableOp0Adam/Hidden_layer_3/kernel/m/Read/ReadVariableOp.Adam/Hidden_layer_3/bias/m/Read/ReadVariableOp.Adam/Output_layer/kernel/m/Read/ReadVariableOp,Adam/Output_layer/bias/m/Read/ReadVariableOp0Adam/Hidden_layer_0/kernel/v/Read/ReadVariableOp.Adam/Hidden_layer_0/bias/v/Read/ReadVariableOp0Adam/Hidden_layer_1/kernel/v/Read/ReadVariableOp.Adam/Hidden_layer_1/bias/v/Read/ReadVariableOp0Adam/Hidden_layer_2/kernel/v/Read/ReadVariableOp.Adam/Hidden_layer_2/bias/v/Read/ReadVariableOp0Adam/Hidden_layer_3/kernel/v/Read/ReadVariableOp.Adam/Hidden_layer_3/bias/v/Read/ReadVariableOp.Adam/Output_layer/kernel/v/Read/ReadVariableOp,Adam/Output_layer/bias/v/Read/ReadVariableOpConst*6
Tin/
-2+	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *(
f#R!
__inference__traced_save_244265
�	
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameHidden_layer_0/kernelHidden_layer_0/biasHidden_layer_1/kernelHidden_layer_1/biasHidden_layer_2/kernelHidden_layer_2/biasHidden_layer_3/kernelHidden_layer_3/biasOutput_layer/kernelOutput_layer/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1total_2count_2Adam/Hidden_layer_0/kernel/mAdam/Hidden_layer_0/bias/mAdam/Hidden_layer_1/kernel/mAdam/Hidden_layer_1/bias/mAdam/Hidden_layer_2/kernel/mAdam/Hidden_layer_2/bias/mAdam/Hidden_layer_3/kernel/mAdam/Hidden_layer_3/bias/mAdam/Output_layer/kernel/mAdam/Output_layer/bias/mAdam/Hidden_layer_0/kernel/vAdam/Hidden_layer_0/bias/vAdam/Hidden_layer_1/kernel/vAdam/Hidden_layer_1/bias/vAdam/Hidden_layer_2/kernel/vAdam/Hidden_layer_2/bias/vAdam/Hidden_layer_3/kernel/vAdam/Hidden_layer_3/bias/vAdam/Output_layer/kernel/vAdam/Output_layer/bias/v*5
Tin.
,2**
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *+
f&R$
"__inference__traced_restore_244398��
�	
�
H__inference_Output_layer_layer_call_and_return_conditional_losses_243441

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:P*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAddU
EluEluBiasAdd:output:0*
T0*'
_output_shapes
:���������2
Elu�
IdentityIdentityElu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������P::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������P
 
_user_specified_nameinputs
�
�
J__inference_Hidden_layer_0_layer_call_and_return_conditional_losses_243315

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�7Hidden_layer_0/kernel/Regularizer/Square/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:P*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:P*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P2	
BiasAddU
EluEluBiasAdd:output:0*
T0*'
_output_shapes
:���������P2
Elu�
7Hidden_layer_0/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:P*
dtype029
7Hidden_layer_0/kernel/Regularizer/Square/ReadVariableOp�
(Hidden_layer_0/kernel/Regularizer/SquareSquare?Hidden_layer_0/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:P2*
(Hidden_layer_0/kernel/Regularizer/Square�
'Hidden_layer_0/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2)
'Hidden_layer_0/kernel/Regularizer/Const�
%Hidden_layer_0/kernel/Regularizer/SumSum,Hidden_layer_0/kernel/Regularizer/Square:y:00Hidden_layer_0/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2'
%Hidden_layer_0/kernel/Regularizer/Sum�
'Hidden_layer_0/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2)
'Hidden_layer_0/kernel/Regularizer/mul/x�
%Hidden_layer_0/kernel/Regularizer/mulMul0Hidden_layer_0/kernel/Regularizer/mul/x:output:0.Hidden_layer_0/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2'
%Hidden_layer_0/kernel/Regularizer/mul�
IdentityIdentityElu:activations:0^BiasAdd/ReadVariableOp8^Hidden_layer_0/kernel/Regularizer/Square/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������P2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2r
7Hidden_layer_0/kernel/Regularizer/Square/ReadVariableOp7Hidden_layer_0/kernel/Regularizer/Square/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
J__inference_Hidden_layer_1_layer_call_and_return_conditional_losses_243348

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�7Hidden_layer_1/kernel/Regularizer/Square/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:PP*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:P*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P2	
BiasAddU
EluEluBiasAdd:output:0*
T0*'
_output_shapes
:���������P2
Elu�
7Hidden_layer_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:PP*
dtype029
7Hidden_layer_1/kernel/Regularizer/Square/ReadVariableOp�
(Hidden_layer_1/kernel/Regularizer/SquareSquare?Hidden_layer_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:PP2*
(Hidden_layer_1/kernel/Regularizer/Square�
'Hidden_layer_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2)
'Hidden_layer_1/kernel/Regularizer/Const�
%Hidden_layer_1/kernel/Regularizer/SumSum,Hidden_layer_1/kernel/Regularizer/Square:y:00Hidden_layer_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2'
%Hidden_layer_1/kernel/Regularizer/Sum�
'Hidden_layer_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2)
'Hidden_layer_1/kernel/Regularizer/mul/x�
%Hidden_layer_1/kernel/Regularizer/mulMul0Hidden_layer_1/kernel/Regularizer/mul/x:output:0.Hidden_layer_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2'
%Hidden_layer_1/kernel/Regularizer/mul�
IdentityIdentityElu:activations:0^BiasAdd/ReadVariableOp8^Hidden_layer_1/kernel/Regularizer/Square/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������P2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������P::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2r
7Hidden_layer_1/kernel/Regularizer/Square/ReadVariableOp7Hidden_layer_1/kernel/Regularizer/Square/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������P
 
_user_specified_nameinputs
�I
�
F__inference_sequential_layer_call_and_return_conditional_losses_243482
input_layer
hidden_layer_0_243326
hidden_layer_0_243328
hidden_layer_1_243359
hidden_layer_1_243361
hidden_layer_2_243392
hidden_layer_2_243394
hidden_layer_3_243425
hidden_layer_3_243427
output_layer_243452
output_layer_243454
identity��&Hidden_layer_0/StatefulPartitionedCall�7Hidden_layer_0/kernel/Regularizer/Square/ReadVariableOp�&Hidden_layer_1/StatefulPartitionedCall�7Hidden_layer_1/kernel/Regularizer/Square/ReadVariableOp�&Hidden_layer_2/StatefulPartitionedCall�7Hidden_layer_2/kernel/Regularizer/Square/ReadVariableOp�&Hidden_layer_3/StatefulPartitionedCall�7Hidden_layer_3/kernel/Regularizer/Square/ReadVariableOp�$Output_layer/StatefulPartitionedCall�
&Hidden_layer_0/StatefulPartitionedCallStatefulPartitionedCallinput_layerhidden_layer_0_243326hidden_layer_0_243328*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������P*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_Hidden_layer_0_layer_call_and_return_conditional_losses_2433152(
&Hidden_layer_0/StatefulPartitionedCall�
&Hidden_layer_1/StatefulPartitionedCallStatefulPartitionedCall/Hidden_layer_0/StatefulPartitionedCall:output:0hidden_layer_1_243359hidden_layer_1_243361*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������P*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_Hidden_layer_1_layer_call_and_return_conditional_losses_2433482(
&Hidden_layer_1/StatefulPartitionedCall�
&Hidden_layer_2/StatefulPartitionedCallStatefulPartitionedCall/Hidden_layer_1/StatefulPartitionedCall:output:0hidden_layer_2_243392hidden_layer_2_243394*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������P*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_Hidden_layer_2_layer_call_and_return_conditional_losses_2433812(
&Hidden_layer_2/StatefulPartitionedCall�
&Hidden_layer_3/StatefulPartitionedCallStatefulPartitionedCall/Hidden_layer_2/StatefulPartitionedCall:output:0hidden_layer_3_243425hidden_layer_3_243427*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������P*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_Hidden_layer_3_layer_call_and_return_conditional_losses_2434142(
&Hidden_layer_3/StatefulPartitionedCall�
$Output_layer/StatefulPartitionedCallStatefulPartitionedCall/Hidden_layer_3/StatefulPartitionedCall:output:0output_layer_243452output_layer_243454*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_Output_layer_layer_call_and_return_conditional_losses_2434412&
$Output_layer/StatefulPartitionedCall�
7Hidden_layer_0/kernel/Regularizer/Square/ReadVariableOpReadVariableOphidden_layer_0_243326*
_output_shapes

:P*
dtype029
7Hidden_layer_0/kernel/Regularizer/Square/ReadVariableOp�
(Hidden_layer_0/kernel/Regularizer/SquareSquare?Hidden_layer_0/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:P2*
(Hidden_layer_0/kernel/Regularizer/Square�
'Hidden_layer_0/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2)
'Hidden_layer_0/kernel/Regularizer/Const�
%Hidden_layer_0/kernel/Regularizer/SumSum,Hidden_layer_0/kernel/Regularizer/Square:y:00Hidden_layer_0/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2'
%Hidden_layer_0/kernel/Regularizer/Sum�
'Hidden_layer_0/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2)
'Hidden_layer_0/kernel/Regularizer/mul/x�
%Hidden_layer_0/kernel/Regularizer/mulMul0Hidden_layer_0/kernel/Regularizer/mul/x:output:0.Hidden_layer_0/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2'
%Hidden_layer_0/kernel/Regularizer/mul�
7Hidden_layer_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOphidden_layer_1_243359*
_output_shapes

:PP*
dtype029
7Hidden_layer_1/kernel/Regularizer/Square/ReadVariableOp�
(Hidden_layer_1/kernel/Regularizer/SquareSquare?Hidden_layer_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:PP2*
(Hidden_layer_1/kernel/Regularizer/Square�
'Hidden_layer_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2)
'Hidden_layer_1/kernel/Regularizer/Const�
%Hidden_layer_1/kernel/Regularizer/SumSum,Hidden_layer_1/kernel/Regularizer/Square:y:00Hidden_layer_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2'
%Hidden_layer_1/kernel/Regularizer/Sum�
'Hidden_layer_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2)
'Hidden_layer_1/kernel/Regularizer/mul/x�
%Hidden_layer_1/kernel/Regularizer/mulMul0Hidden_layer_1/kernel/Regularizer/mul/x:output:0.Hidden_layer_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2'
%Hidden_layer_1/kernel/Regularizer/mul�
7Hidden_layer_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOphidden_layer_2_243392*
_output_shapes

:PP*
dtype029
7Hidden_layer_2/kernel/Regularizer/Square/ReadVariableOp�
(Hidden_layer_2/kernel/Regularizer/SquareSquare?Hidden_layer_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:PP2*
(Hidden_layer_2/kernel/Regularizer/Square�
'Hidden_layer_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2)
'Hidden_layer_2/kernel/Regularizer/Const�
%Hidden_layer_2/kernel/Regularizer/SumSum,Hidden_layer_2/kernel/Regularizer/Square:y:00Hidden_layer_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2'
%Hidden_layer_2/kernel/Regularizer/Sum�
'Hidden_layer_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2)
'Hidden_layer_2/kernel/Regularizer/mul/x�
%Hidden_layer_2/kernel/Regularizer/mulMul0Hidden_layer_2/kernel/Regularizer/mul/x:output:0.Hidden_layer_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2'
%Hidden_layer_2/kernel/Regularizer/mul�
7Hidden_layer_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOphidden_layer_3_243425*
_output_shapes

:PP*
dtype029
7Hidden_layer_3/kernel/Regularizer/Square/ReadVariableOp�
(Hidden_layer_3/kernel/Regularizer/SquareSquare?Hidden_layer_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:PP2*
(Hidden_layer_3/kernel/Regularizer/Square�
'Hidden_layer_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2)
'Hidden_layer_3/kernel/Regularizer/Const�
%Hidden_layer_3/kernel/Regularizer/SumSum,Hidden_layer_3/kernel/Regularizer/Square:y:00Hidden_layer_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2'
%Hidden_layer_3/kernel/Regularizer/Sum�
'Hidden_layer_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2)
'Hidden_layer_3/kernel/Regularizer/mul/x�
%Hidden_layer_3/kernel/Regularizer/mulMul0Hidden_layer_3/kernel/Regularizer/mul/x:output:0.Hidden_layer_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2'
%Hidden_layer_3/kernel/Regularizer/mul�
IdentityIdentity-Output_layer/StatefulPartitionedCall:output:0'^Hidden_layer_0/StatefulPartitionedCall8^Hidden_layer_0/kernel/Regularizer/Square/ReadVariableOp'^Hidden_layer_1/StatefulPartitionedCall8^Hidden_layer_1/kernel/Regularizer/Square/ReadVariableOp'^Hidden_layer_2/StatefulPartitionedCall8^Hidden_layer_2/kernel/Regularizer/Square/ReadVariableOp'^Hidden_layer_3/StatefulPartitionedCall8^Hidden_layer_3/kernel/Regularizer/Square/ReadVariableOp%^Output_layer/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:���������::::::::::2P
&Hidden_layer_0/StatefulPartitionedCall&Hidden_layer_0/StatefulPartitionedCall2r
7Hidden_layer_0/kernel/Regularizer/Square/ReadVariableOp7Hidden_layer_0/kernel/Regularizer/Square/ReadVariableOp2P
&Hidden_layer_1/StatefulPartitionedCall&Hidden_layer_1/StatefulPartitionedCall2r
7Hidden_layer_1/kernel/Regularizer/Square/ReadVariableOp7Hidden_layer_1/kernel/Regularizer/Square/ReadVariableOp2P
&Hidden_layer_2/StatefulPartitionedCall&Hidden_layer_2/StatefulPartitionedCall2r
7Hidden_layer_2/kernel/Regularizer/Square/ReadVariableOp7Hidden_layer_2/kernel/Regularizer/Square/ReadVariableOp2P
&Hidden_layer_3/StatefulPartitionedCall&Hidden_layer_3/StatefulPartitionedCall2r
7Hidden_layer_3/kernel/Regularizer/Square/ReadVariableOp7Hidden_layer_3/kernel/Regularizer/Square/ReadVariableOp2L
$Output_layer/StatefulPartitionedCall$Output_layer/StatefulPartitionedCall:T P
'
_output_shapes
:���������
%
_user_specified_nameInput_layer
�W
�
__inference__traced_save_244265
file_prefix4
0savev2_hidden_layer_0_kernel_read_readvariableop2
.savev2_hidden_layer_0_bias_read_readvariableop4
0savev2_hidden_layer_1_kernel_read_readvariableop2
.savev2_hidden_layer_1_bias_read_readvariableop4
0savev2_hidden_layer_2_kernel_read_readvariableop2
.savev2_hidden_layer_2_bias_read_readvariableop4
0savev2_hidden_layer_3_kernel_read_readvariableop2
.savev2_hidden_layer_3_bias_read_readvariableop2
.savev2_output_layer_kernel_read_readvariableop0
,savev2_output_layer_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop&
"savev2_total_2_read_readvariableop&
"savev2_count_2_read_readvariableop;
7savev2_adam_hidden_layer_0_kernel_m_read_readvariableop9
5savev2_adam_hidden_layer_0_bias_m_read_readvariableop;
7savev2_adam_hidden_layer_1_kernel_m_read_readvariableop9
5savev2_adam_hidden_layer_1_bias_m_read_readvariableop;
7savev2_adam_hidden_layer_2_kernel_m_read_readvariableop9
5savev2_adam_hidden_layer_2_bias_m_read_readvariableop;
7savev2_adam_hidden_layer_3_kernel_m_read_readvariableop9
5savev2_adam_hidden_layer_3_bias_m_read_readvariableop9
5savev2_adam_output_layer_kernel_m_read_readvariableop7
3savev2_adam_output_layer_bias_m_read_readvariableop;
7savev2_adam_hidden_layer_0_kernel_v_read_readvariableop9
5savev2_adam_hidden_layer_0_bias_v_read_readvariableop;
7savev2_adam_hidden_layer_1_kernel_v_read_readvariableop9
5savev2_adam_hidden_layer_1_bias_v_read_readvariableop;
7savev2_adam_hidden_layer_2_kernel_v_read_readvariableop9
5savev2_adam_hidden_layer_2_bias_v_read_readvariableop;
7savev2_adam_hidden_layer_3_kernel_v_read_readvariableop9
5savev2_adam_hidden_layer_3_bias_v_read_readvariableop9
5savev2_adam_output_layer_kernel_v_read_readvariableop7
3savev2_adam_output_layer_bias_v_read_readvariableop
savev2_const

identity_1��MergeV2Checkpoints�
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard�
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename�
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:**
dtype0*�
value�B�*B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:**
dtype0*g
value^B\*B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices�
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:00savev2_hidden_layer_0_kernel_read_readvariableop.savev2_hidden_layer_0_bias_read_readvariableop0savev2_hidden_layer_1_kernel_read_readvariableop.savev2_hidden_layer_1_bias_read_readvariableop0savev2_hidden_layer_2_kernel_read_readvariableop.savev2_hidden_layer_2_bias_read_readvariableop0savev2_hidden_layer_3_kernel_read_readvariableop.savev2_hidden_layer_3_bias_read_readvariableop.savev2_output_layer_kernel_read_readvariableop,savev2_output_layer_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop"savev2_total_2_read_readvariableop"savev2_count_2_read_readvariableop7savev2_adam_hidden_layer_0_kernel_m_read_readvariableop5savev2_adam_hidden_layer_0_bias_m_read_readvariableop7savev2_adam_hidden_layer_1_kernel_m_read_readvariableop5savev2_adam_hidden_layer_1_bias_m_read_readvariableop7savev2_adam_hidden_layer_2_kernel_m_read_readvariableop5savev2_adam_hidden_layer_2_bias_m_read_readvariableop7savev2_adam_hidden_layer_3_kernel_m_read_readvariableop5savev2_adam_hidden_layer_3_bias_m_read_readvariableop5savev2_adam_output_layer_kernel_m_read_readvariableop3savev2_adam_output_layer_bias_m_read_readvariableop7savev2_adam_hidden_layer_0_kernel_v_read_readvariableop5savev2_adam_hidden_layer_0_bias_v_read_readvariableop7savev2_adam_hidden_layer_1_kernel_v_read_readvariableop5savev2_adam_hidden_layer_1_bias_v_read_readvariableop7savev2_adam_hidden_layer_2_kernel_v_read_readvariableop5savev2_adam_hidden_layer_2_bias_v_read_readvariableop7savev2_adam_hidden_layer_3_kernel_v_read_readvariableop5savev2_adam_hidden_layer_3_bias_v_read_readvariableop5savev2_adam_output_layer_kernel_v_read_readvariableop3savev2_adam_output_layer_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *8
dtypes.
,2*	2
SaveV2�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*�
_input_shapes�
�: :P:P:PP:P:PP:P:PP:P:P:: : : : : : : : : : : :P:P:PP:P:PP:P:PP:P:P::P:P:PP:P:PP:P:PP:P:P:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:P: 

_output_shapes
:P:$ 

_output_shapes

:PP: 

_output_shapes
:P:$ 

_output_shapes

:PP: 

_output_shapes
:P:$ 

_output_shapes

:PP: 

_output_shapes
:P:$	 

_output_shapes

:P: 


_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:P: 

_output_shapes
:P:$ 

_output_shapes

:PP: 

_output_shapes
:P:$ 

_output_shapes

:PP: 

_output_shapes
:P:$ 

_output_shapes

:PP: 

_output_shapes
:P:$ 

_output_shapes

:P: 

_output_shapes
::$  

_output_shapes

:P: !

_output_shapes
:P:$" 

_output_shapes

:PP: #

_output_shapes
:P:$$ 

_output_shapes

:PP: %

_output_shapes
:P:$& 

_output_shapes

:PP: '

_output_shapes
:P:$( 

_output_shapes

:P: )

_output_shapes
::*

_output_shapes
: 
�
�
+__inference_sequential_layer_call_fn_243692
input_layer
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_layerunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_2436692
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:���������::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
'
_output_shapes
:���������
%
_user_specified_nameInput_layer
�b
�	
F__inference_sequential_layer_call_and_return_conditional_losses_243877

inputs1
-hidden_layer_0_matmul_readvariableop_resource2
.hidden_layer_0_biasadd_readvariableop_resource1
-hidden_layer_1_matmul_readvariableop_resource2
.hidden_layer_1_biasadd_readvariableop_resource1
-hidden_layer_2_matmul_readvariableop_resource2
.hidden_layer_2_biasadd_readvariableop_resource1
-hidden_layer_3_matmul_readvariableop_resource2
.hidden_layer_3_biasadd_readvariableop_resource/
+output_layer_matmul_readvariableop_resource0
,output_layer_biasadd_readvariableop_resource
identity��%Hidden_layer_0/BiasAdd/ReadVariableOp�$Hidden_layer_0/MatMul/ReadVariableOp�7Hidden_layer_0/kernel/Regularizer/Square/ReadVariableOp�%Hidden_layer_1/BiasAdd/ReadVariableOp�$Hidden_layer_1/MatMul/ReadVariableOp�7Hidden_layer_1/kernel/Regularizer/Square/ReadVariableOp�%Hidden_layer_2/BiasAdd/ReadVariableOp�$Hidden_layer_2/MatMul/ReadVariableOp�7Hidden_layer_2/kernel/Regularizer/Square/ReadVariableOp�%Hidden_layer_3/BiasAdd/ReadVariableOp�$Hidden_layer_3/MatMul/ReadVariableOp�7Hidden_layer_3/kernel/Regularizer/Square/ReadVariableOp�#Output_layer/BiasAdd/ReadVariableOp�"Output_layer/MatMul/ReadVariableOp�
$Hidden_layer_0/MatMul/ReadVariableOpReadVariableOp-hidden_layer_0_matmul_readvariableop_resource*
_output_shapes

:P*
dtype02&
$Hidden_layer_0/MatMul/ReadVariableOp�
Hidden_layer_0/MatMulMatMulinputs,Hidden_layer_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P2
Hidden_layer_0/MatMul�
%Hidden_layer_0/BiasAdd/ReadVariableOpReadVariableOp.hidden_layer_0_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype02'
%Hidden_layer_0/BiasAdd/ReadVariableOp�
Hidden_layer_0/BiasAddBiasAddHidden_layer_0/MatMul:product:0-Hidden_layer_0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P2
Hidden_layer_0/BiasAdd�
Hidden_layer_0/EluEluHidden_layer_0/BiasAdd:output:0*
T0*'
_output_shapes
:���������P2
Hidden_layer_0/Elu�
$Hidden_layer_1/MatMul/ReadVariableOpReadVariableOp-hidden_layer_1_matmul_readvariableop_resource*
_output_shapes

:PP*
dtype02&
$Hidden_layer_1/MatMul/ReadVariableOp�
Hidden_layer_1/MatMulMatMul Hidden_layer_0/Elu:activations:0,Hidden_layer_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P2
Hidden_layer_1/MatMul�
%Hidden_layer_1/BiasAdd/ReadVariableOpReadVariableOp.hidden_layer_1_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype02'
%Hidden_layer_1/BiasAdd/ReadVariableOp�
Hidden_layer_1/BiasAddBiasAddHidden_layer_1/MatMul:product:0-Hidden_layer_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P2
Hidden_layer_1/BiasAdd�
Hidden_layer_1/EluEluHidden_layer_1/BiasAdd:output:0*
T0*'
_output_shapes
:���������P2
Hidden_layer_1/Elu�
$Hidden_layer_2/MatMul/ReadVariableOpReadVariableOp-hidden_layer_2_matmul_readvariableop_resource*
_output_shapes

:PP*
dtype02&
$Hidden_layer_2/MatMul/ReadVariableOp�
Hidden_layer_2/MatMulMatMul Hidden_layer_1/Elu:activations:0,Hidden_layer_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P2
Hidden_layer_2/MatMul�
%Hidden_layer_2/BiasAdd/ReadVariableOpReadVariableOp.hidden_layer_2_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype02'
%Hidden_layer_2/BiasAdd/ReadVariableOp�
Hidden_layer_2/BiasAddBiasAddHidden_layer_2/MatMul:product:0-Hidden_layer_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P2
Hidden_layer_2/BiasAdd�
Hidden_layer_2/EluEluHidden_layer_2/BiasAdd:output:0*
T0*'
_output_shapes
:���������P2
Hidden_layer_2/Elu�
$Hidden_layer_3/MatMul/ReadVariableOpReadVariableOp-hidden_layer_3_matmul_readvariableop_resource*
_output_shapes

:PP*
dtype02&
$Hidden_layer_3/MatMul/ReadVariableOp�
Hidden_layer_3/MatMulMatMul Hidden_layer_2/Elu:activations:0,Hidden_layer_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P2
Hidden_layer_3/MatMul�
%Hidden_layer_3/BiasAdd/ReadVariableOpReadVariableOp.hidden_layer_3_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype02'
%Hidden_layer_3/BiasAdd/ReadVariableOp�
Hidden_layer_3/BiasAddBiasAddHidden_layer_3/MatMul:product:0-Hidden_layer_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P2
Hidden_layer_3/BiasAdd�
Hidden_layer_3/EluEluHidden_layer_3/BiasAdd:output:0*
T0*'
_output_shapes
:���������P2
Hidden_layer_3/Elu�
"Output_layer/MatMul/ReadVariableOpReadVariableOp+output_layer_matmul_readvariableop_resource*
_output_shapes

:P*
dtype02$
"Output_layer/MatMul/ReadVariableOp�
Output_layer/MatMulMatMul Hidden_layer_3/Elu:activations:0*Output_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
Output_layer/MatMul�
#Output_layer/BiasAdd/ReadVariableOpReadVariableOp,output_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02%
#Output_layer/BiasAdd/ReadVariableOp�
Output_layer/BiasAddBiasAddOutput_layer/MatMul:product:0+Output_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
Output_layer/BiasAdd|
Output_layer/EluEluOutput_layer/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
Output_layer/Elu�
7Hidden_layer_0/kernel/Regularizer/Square/ReadVariableOpReadVariableOp-hidden_layer_0_matmul_readvariableop_resource*
_output_shapes

:P*
dtype029
7Hidden_layer_0/kernel/Regularizer/Square/ReadVariableOp�
(Hidden_layer_0/kernel/Regularizer/SquareSquare?Hidden_layer_0/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:P2*
(Hidden_layer_0/kernel/Regularizer/Square�
'Hidden_layer_0/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2)
'Hidden_layer_0/kernel/Regularizer/Const�
%Hidden_layer_0/kernel/Regularizer/SumSum,Hidden_layer_0/kernel/Regularizer/Square:y:00Hidden_layer_0/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2'
%Hidden_layer_0/kernel/Regularizer/Sum�
'Hidden_layer_0/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2)
'Hidden_layer_0/kernel/Regularizer/mul/x�
%Hidden_layer_0/kernel/Regularizer/mulMul0Hidden_layer_0/kernel/Regularizer/mul/x:output:0.Hidden_layer_0/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2'
%Hidden_layer_0/kernel/Regularizer/mul�
7Hidden_layer_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp-hidden_layer_1_matmul_readvariableop_resource*
_output_shapes

:PP*
dtype029
7Hidden_layer_1/kernel/Regularizer/Square/ReadVariableOp�
(Hidden_layer_1/kernel/Regularizer/SquareSquare?Hidden_layer_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:PP2*
(Hidden_layer_1/kernel/Regularizer/Square�
'Hidden_layer_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2)
'Hidden_layer_1/kernel/Regularizer/Const�
%Hidden_layer_1/kernel/Regularizer/SumSum,Hidden_layer_1/kernel/Regularizer/Square:y:00Hidden_layer_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2'
%Hidden_layer_1/kernel/Regularizer/Sum�
'Hidden_layer_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2)
'Hidden_layer_1/kernel/Regularizer/mul/x�
%Hidden_layer_1/kernel/Regularizer/mulMul0Hidden_layer_1/kernel/Regularizer/mul/x:output:0.Hidden_layer_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2'
%Hidden_layer_1/kernel/Regularizer/mul�
7Hidden_layer_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp-hidden_layer_2_matmul_readvariableop_resource*
_output_shapes

:PP*
dtype029
7Hidden_layer_2/kernel/Regularizer/Square/ReadVariableOp�
(Hidden_layer_2/kernel/Regularizer/SquareSquare?Hidden_layer_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:PP2*
(Hidden_layer_2/kernel/Regularizer/Square�
'Hidden_layer_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2)
'Hidden_layer_2/kernel/Regularizer/Const�
%Hidden_layer_2/kernel/Regularizer/SumSum,Hidden_layer_2/kernel/Regularizer/Square:y:00Hidden_layer_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2'
%Hidden_layer_2/kernel/Regularizer/Sum�
'Hidden_layer_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2)
'Hidden_layer_2/kernel/Regularizer/mul/x�
%Hidden_layer_2/kernel/Regularizer/mulMul0Hidden_layer_2/kernel/Regularizer/mul/x:output:0.Hidden_layer_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2'
%Hidden_layer_2/kernel/Regularizer/mul�
7Hidden_layer_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOp-hidden_layer_3_matmul_readvariableop_resource*
_output_shapes

:PP*
dtype029
7Hidden_layer_3/kernel/Regularizer/Square/ReadVariableOp�
(Hidden_layer_3/kernel/Regularizer/SquareSquare?Hidden_layer_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:PP2*
(Hidden_layer_3/kernel/Regularizer/Square�
'Hidden_layer_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2)
'Hidden_layer_3/kernel/Regularizer/Const�
%Hidden_layer_3/kernel/Regularizer/SumSum,Hidden_layer_3/kernel/Regularizer/Square:y:00Hidden_layer_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2'
%Hidden_layer_3/kernel/Regularizer/Sum�
'Hidden_layer_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2)
'Hidden_layer_3/kernel/Regularizer/mul/x�
%Hidden_layer_3/kernel/Regularizer/mulMul0Hidden_layer_3/kernel/Regularizer/mul/x:output:0.Hidden_layer_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2'
%Hidden_layer_3/kernel/Regularizer/mul�
IdentityIdentityOutput_layer/Elu:activations:0&^Hidden_layer_0/BiasAdd/ReadVariableOp%^Hidden_layer_0/MatMul/ReadVariableOp8^Hidden_layer_0/kernel/Regularizer/Square/ReadVariableOp&^Hidden_layer_1/BiasAdd/ReadVariableOp%^Hidden_layer_1/MatMul/ReadVariableOp8^Hidden_layer_1/kernel/Regularizer/Square/ReadVariableOp&^Hidden_layer_2/BiasAdd/ReadVariableOp%^Hidden_layer_2/MatMul/ReadVariableOp8^Hidden_layer_2/kernel/Regularizer/Square/ReadVariableOp&^Hidden_layer_3/BiasAdd/ReadVariableOp%^Hidden_layer_3/MatMul/ReadVariableOp8^Hidden_layer_3/kernel/Regularizer/Square/ReadVariableOp$^Output_layer/BiasAdd/ReadVariableOp#^Output_layer/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:���������::::::::::2N
%Hidden_layer_0/BiasAdd/ReadVariableOp%Hidden_layer_0/BiasAdd/ReadVariableOp2L
$Hidden_layer_0/MatMul/ReadVariableOp$Hidden_layer_0/MatMul/ReadVariableOp2r
7Hidden_layer_0/kernel/Regularizer/Square/ReadVariableOp7Hidden_layer_0/kernel/Regularizer/Square/ReadVariableOp2N
%Hidden_layer_1/BiasAdd/ReadVariableOp%Hidden_layer_1/BiasAdd/ReadVariableOp2L
$Hidden_layer_1/MatMul/ReadVariableOp$Hidden_layer_1/MatMul/ReadVariableOp2r
7Hidden_layer_1/kernel/Regularizer/Square/ReadVariableOp7Hidden_layer_1/kernel/Regularizer/Square/ReadVariableOp2N
%Hidden_layer_2/BiasAdd/ReadVariableOp%Hidden_layer_2/BiasAdd/ReadVariableOp2L
$Hidden_layer_2/MatMul/ReadVariableOp$Hidden_layer_2/MatMul/ReadVariableOp2r
7Hidden_layer_2/kernel/Regularizer/Square/ReadVariableOp7Hidden_layer_2/kernel/Regularizer/Square/ReadVariableOp2N
%Hidden_layer_3/BiasAdd/ReadVariableOp%Hidden_layer_3/BiasAdd/ReadVariableOp2L
$Hidden_layer_3/MatMul/ReadVariableOp$Hidden_layer_3/MatMul/ReadVariableOp2r
7Hidden_layer_3/kernel/Regularizer/Square/ReadVariableOp7Hidden_layer_3/kernel/Regularizer/Square/ReadVariableOp2J
#Output_layer/BiasAdd/ReadVariableOp#Output_layer/BiasAdd/ReadVariableOp2H
"Output_layer/MatMul/ReadVariableOp"Output_layer/MatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
J__inference_Hidden_layer_3_layer_call_and_return_conditional_losses_244046

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�7Hidden_layer_3/kernel/Regularizer/Square/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:PP*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:P*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P2	
BiasAddU
EluEluBiasAdd:output:0*
T0*'
_output_shapes
:���������P2
Elu�
7Hidden_layer_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:PP*
dtype029
7Hidden_layer_3/kernel/Regularizer/Square/ReadVariableOp�
(Hidden_layer_3/kernel/Regularizer/SquareSquare?Hidden_layer_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:PP2*
(Hidden_layer_3/kernel/Regularizer/Square�
'Hidden_layer_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2)
'Hidden_layer_3/kernel/Regularizer/Const�
%Hidden_layer_3/kernel/Regularizer/SumSum,Hidden_layer_3/kernel/Regularizer/Square:y:00Hidden_layer_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2'
%Hidden_layer_3/kernel/Regularizer/Sum�
'Hidden_layer_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2)
'Hidden_layer_3/kernel/Regularizer/mul/x�
%Hidden_layer_3/kernel/Regularizer/mulMul0Hidden_layer_3/kernel/Regularizer/mul/x:output:0.Hidden_layer_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2'
%Hidden_layer_3/kernel/Regularizer/mul�
IdentityIdentityElu:activations:0^BiasAdd/ReadVariableOp8^Hidden_layer_3/kernel/Regularizer/Square/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������P2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������P::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2r
7Hidden_layer_3/kernel/Regularizer/Square/ReadVariableOp7Hidden_layer_3/kernel/Regularizer/Square/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������P
 
_user_specified_nameinputs
�
�
__inference_loss_fn_0_244086D
@hidden_layer_0_kernel_regularizer_square_readvariableop_resource
identity��7Hidden_layer_0/kernel/Regularizer/Square/ReadVariableOp�
7Hidden_layer_0/kernel/Regularizer/Square/ReadVariableOpReadVariableOp@hidden_layer_0_kernel_regularizer_square_readvariableop_resource*
_output_shapes

:P*
dtype029
7Hidden_layer_0/kernel/Regularizer/Square/ReadVariableOp�
(Hidden_layer_0/kernel/Regularizer/SquareSquare?Hidden_layer_0/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:P2*
(Hidden_layer_0/kernel/Regularizer/Square�
'Hidden_layer_0/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2)
'Hidden_layer_0/kernel/Regularizer/Const�
%Hidden_layer_0/kernel/Regularizer/SumSum,Hidden_layer_0/kernel/Regularizer/Square:y:00Hidden_layer_0/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2'
%Hidden_layer_0/kernel/Regularizer/Sum�
'Hidden_layer_0/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2)
'Hidden_layer_0/kernel/Regularizer/mul/x�
%Hidden_layer_0/kernel/Regularizer/mulMul0Hidden_layer_0/kernel/Regularizer/mul/x:output:0.Hidden_layer_0/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2'
%Hidden_layer_0/kernel/Regularizer/mul�
IdentityIdentity)Hidden_layer_0/kernel/Regularizer/mul:z:08^Hidden_layer_0/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:2r
7Hidden_layer_0/kernel/Regularizer/Square/ReadVariableOp7Hidden_layer_0/kernel/Regularizer/Square/ReadVariableOp
�
�
"__inference__traced_restore_244398
file_prefix*
&assignvariableop_hidden_layer_0_kernel*
&assignvariableop_1_hidden_layer_0_bias,
(assignvariableop_2_hidden_layer_1_kernel*
&assignvariableop_3_hidden_layer_1_bias,
(assignvariableop_4_hidden_layer_2_kernel*
&assignvariableop_5_hidden_layer_2_bias,
(assignvariableop_6_hidden_layer_3_kernel*
&assignvariableop_7_hidden_layer_3_bias*
&assignvariableop_8_output_layer_kernel(
$assignvariableop_9_output_layer_bias!
assignvariableop_10_adam_iter#
assignvariableop_11_adam_beta_1#
assignvariableop_12_adam_beta_2"
assignvariableop_13_adam_decay*
&assignvariableop_14_adam_learning_rate
assignvariableop_15_total
assignvariableop_16_count
assignvariableop_17_total_1
assignvariableop_18_count_1
assignvariableop_19_total_2
assignvariableop_20_count_24
0assignvariableop_21_adam_hidden_layer_0_kernel_m2
.assignvariableop_22_adam_hidden_layer_0_bias_m4
0assignvariableop_23_adam_hidden_layer_1_kernel_m2
.assignvariableop_24_adam_hidden_layer_1_bias_m4
0assignvariableop_25_adam_hidden_layer_2_kernel_m2
.assignvariableop_26_adam_hidden_layer_2_bias_m4
0assignvariableop_27_adam_hidden_layer_3_kernel_m2
.assignvariableop_28_adam_hidden_layer_3_bias_m2
.assignvariableop_29_adam_output_layer_kernel_m0
,assignvariableop_30_adam_output_layer_bias_m4
0assignvariableop_31_adam_hidden_layer_0_kernel_v2
.assignvariableop_32_adam_hidden_layer_0_bias_v4
0assignvariableop_33_adam_hidden_layer_1_kernel_v2
.assignvariableop_34_adam_hidden_layer_1_bias_v4
0assignvariableop_35_adam_hidden_layer_2_kernel_v2
.assignvariableop_36_adam_hidden_layer_2_bias_v4
0assignvariableop_37_adam_hidden_layer_3_kernel_v2
.assignvariableop_38_adam_hidden_layer_3_bias_v2
.assignvariableop_39_adam_output_layer_kernel_v0
,assignvariableop_40_adam_output_layer_bias_v
identity_42��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_34�AssignVariableOp_35�AssignVariableOp_36�AssignVariableOp_37�AssignVariableOp_38�AssignVariableOp_39�AssignVariableOp_4�AssignVariableOp_40�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:**
dtype0*�
value�B�*B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:**
dtype0*g
value^B\*B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices�
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::::*8
dtypes.
,2*	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity�
AssignVariableOpAssignVariableOp&assignvariableop_hidden_layer_0_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1�
AssignVariableOp_1AssignVariableOp&assignvariableop_1_hidden_layer_0_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2�
AssignVariableOp_2AssignVariableOp(assignvariableop_2_hidden_layer_1_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3�
AssignVariableOp_3AssignVariableOp&assignvariableop_3_hidden_layer_1_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4�
AssignVariableOp_4AssignVariableOp(assignvariableop_4_hidden_layer_2_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5�
AssignVariableOp_5AssignVariableOp&assignvariableop_5_hidden_layer_2_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6�
AssignVariableOp_6AssignVariableOp(assignvariableop_6_hidden_layer_3_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7�
AssignVariableOp_7AssignVariableOp&assignvariableop_7_hidden_layer_3_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8�
AssignVariableOp_8AssignVariableOp&assignvariableop_8_output_layer_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9�
AssignVariableOp_9AssignVariableOp$assignvariableop_9_output_layer_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_10�
AssignVariableOp_10AssignVariableOpassignvariableop_10_adam_iterIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11�
AssignVariableOp_11AssignVariableOpassignvariableop_11_adam_beta_1Identity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12�
AssignVariableOp_12AssignVariableOpassignvariableop_12_adam_beta_2Identity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13�
AssignVariableOp_13AssignVariableOpassignvariableop_13_adam_decayIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14�
AssignVariableOp_14AssignVariableOp&assignvariableop_14_adam_learning_rateIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15�
AssignVariableOp_15AssignVariableOpassignvariableop_15_totalIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16�
AssignVariableOp_16AssignVariableOpassignvariableop_16_countIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17�
AssignVariableOp_17AssignVariableOpassignvariableop_17_total_1Identity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18�
AssignVariableOp_18AssignVariableOpassignvariableop_18_count_1Identity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19�
AssignVariableOp_19AssignVariableOpassignvariableop_19_total_2Identity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20�
AssignVariableOp_20AssignVariableOpassignvariableop_20_count_2Identity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21�
AssignVariableOp_21AssignVariableOp0assignvariableop_21_adam_hidden_layer_0_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22�
AssignVariableOp_22AssignVariableOp.assignvariableop_22_adam_hidden_layer_0_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23�
AssignVariableOp_23AssignVariableOp0assignvariableop_23_adam_hidden_layer_1_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24�
AssignVariableOp_24AssignVariableOp.assignvariableop_24_adam_hidden_layer_1_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25�
AssignVariableOp_25AssignVariableOp0assignvariableop_25_adam_hidden_layer_2_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26�
AssignVariableOp_26AssignVariableOp.assignvariableop_26_adam_hidden_layer_2_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27�
AssignVariableOp_27AssignVariableOp0assignvariableop_27_adam_hidden_layer_3_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28�
AssignVariableOp_28AssignVariableOp.assignvariableop_28_adam_hidden_layer_3_bias_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29�
AssignVariableOp_29AssignVariableOp.assignvariableop_29_adam_output_layer_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30�
AssignVariableOp_30AssignVariableOp,assignvariableop_30_adam_output_layer_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31�
AssignVariableOp_31AssignVariableOp0assignvariableop_31_adam_hidden_layer_0_kernel_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32�
AssignVariableOp_32AssignVariableOp.assignvariableop_32_adam_hidden_layer_0_bias_vIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33�
AssignVariableOp_33AssignVariableOp0assignvariableop_33_adam_hidden_layer_1_kernel_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34�
AssignVariableOp_34AssignVariableOp.assignvariableop_34_adam_hidden_layer_1_bias_vIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35�
AssignVariableOp_35AssignVariableOp0assignvariableop_35_adam_hidden_layer_2_kernel_vIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36�
AssignVariableOp_36AssignVariableOp.assignvariableop_36_adam_hidden_layer_2_bias_vIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37�
AssignVariableOp_37AssignVariableOp0assignvariableop_37_adam_hidden_layer_3_kernel_vIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38�
AssignVariableOp_38AssignVariableOp.assignvariableop_38_adam_hidden_layer_3_bias_vIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39�
AssignVariableOp_39AssignVariableOp.assignvariableop_39_adam_output_layer_kernel_vIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40�
AssignVariableOp_40AssignVariableOp,assignvariableop_40_adam_output_layer_bias_vIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_409
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp�
Identity_41Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_41�
Identity_42IdentityIdentity_41:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_42"#
identity_42Identity_42:output:0*�
_input_shapes�
�: :::::::::::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
�
J__inference_Hidden_layer_3_layer_call_and_return_conditional_losses_243414

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�7Hidden_layer_3/kernel/Regularizer/Square/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:PP*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:P*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P2	
BiasAddU
EluEluBiasAdd:output:0*
T0*'
_output_shapes
:���������P2
Elu�
7Hidden_layer_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:PP*
dtype029
7Hidden_layer_3/kernel/Regularizer/Square/ReadVariableOp�
(Hidden_layer_3/kernel/Regularizer/SquareSquare?Hidden_layer_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:PP2*
(Hidden_layer_3/kernel/Regularizer/Square�
'Hidden_layer_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2)
'Hidden_layer_3/kernel/Regularizer/Const�
%Hidden_layer_3/kernel/Regularizer/SumSum,Hidden_layer_3/kernel/Regularizer/Square:y:00Hidden_layer_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2'
%Hidden_layer_3/kernel/Regularizer/Sum�
'Hidden_layer_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2)
'Hidden_layer_3/kernel/Regularizer/mul/x�
%Hidden_layer_3/kernel/Regularizer/mulMul0Hidden_layer_3/kernel/Regularizer/mul/x:output:0.Hidden_layer_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2'
%Hidden_layer_3/kernel/Regularizer/mul�
IdentityIdentityElu:activations:0^BiasAdd/ReadVariableOp8^Hidden_layer_3/kernel/Regularizer/Square/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������P2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������P::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2r
7Hidden_layer_3/kernel/Regularizer/Square/ReadVariableOp7Hidden_layer_3/kernel/Regularizer/Square/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������P
 
_user_specified_nameinputs
�
�
+__inference_sequential_layer_call_fn_243927

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_2436692
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:���������::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
J__inference_Hidden_layer_1_layer_call_and_return_conditional_losses_243982

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�7Hidden_layer_1/kernel/Regularizer/Square/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:PP*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:P*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P2	
BiasAddU
EluEluBiasAdd:output:0*
T0*'
_output_shapes
:���������P2
Elu�
7Hidden_layer_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:PP*
dtype029
7Hidden_layer_1/kernel/Regularizer/Square/ReadVariableOp�
(Hidden_layer_1/kernel/Regularizer/SquareSquare?Hidden_layer_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:PP2*
(Hidden_layer_1/kernel/Regularizer/Square�
'Hidden_layer_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2)
'Hidden_layer_1/kernel/Regularizer/Const�
%Hidden_layer_1/kernel/Regularizer/SumSum,Hidden_layer_1/kernel/Regularizer/Square:y:00Hidden_layer_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2'
%Hidden_layer_1/kernel/Regularizer/Sum�
'Hidden_layer_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2)
'Hidden_layer_1/kernel/Regularizer/mul/x�
%Hidden_layer_1/kernel/Regularizer/mulMul0Hidden_layer_1/kernel/Regularizer/mul/x:output:0.Hidden_layer_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2'
%Hidden_layer_1/kernel/Regularizer/mul�
IdentityIdentityElu:activations:0^BiasAdd/ReadVariableOp8^Hidden_layer_1/kernel/Regularizer/Square/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������P2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������P::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2r
7Hidden_layer_1/kernel/Regularizer/Square/ReadVariableOp7Hidden_layer_1/kernel/Regularizer/Square/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������P
 
_user_specified_nameinputs
�
�
-__inference_Output_layer_layer_call_fn_244075

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_Output_layer_layer_call_and_return_conditional_losses_2434412
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������P::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������P
 
_user_specified_nameinputs
�b
�	
F__inference_sequential_layer_call_and_return_conditional_losses_243814

inputs1
-hidden_layer_0_matmul_readvariableop_resource2
.hidden_layer_0_biasadd_readvariableop_resource1
-hidden_layer_1_matmul_readvariableop_resource2
.hidden_layer_1_biasadd_readvariableop_resource1
-hidden_layer_2_matmul_readvariableop_resource2
.hidden_layer_2_biasadd_readvariableop_resource1
-hidden_layer_3_matmul_readvariableop_resource2
.hidden_layer_3_biasadd_readvariableop_resource/
+output_layer_matmul_readvariableop_resource0
,output_layer_biasadd_readvariableop_resource
identity��%Hidden_layer_0/BiasAdd/ReadVariableOp�$Hidden_layer_0/MatMul/ReadVariableOp�7Hidden_layer_0/kernel/Regularizer/Square/ReadVariableOp�%Hidden_layer_1/BiasAdd/ReadVariableOp�$Hidden_layer_1/MatMul/ReadVariableOp�7Hidden_layer_1/kernel/Regularizer/Square/ReadVariableOp�%Hidden_layer_2/BiasAdd/ReadVariableOp�$Hidden_layer_2/MatMul/ReadVariableOp�7Hidden_layer_2/kernel/Regularizer/Square/ReadVariableOp�%Hidden_layer_3/BiasAdd/ReadVariableOp�$Hidden_layer_3/MatMul/ReadVariableOp�7Hidden_layer_3/kernel/Regularizer/Square/ReadVariableOp�#Output_layer/BiasAdd/ReadVariableOp�"Output_layer/MatMul/ReadVariableOp�
$Hidden_layer_0/MatMul/ReadVariableOpReadVariableOp-hidden_layer_0_matmul_readvariableop_resource*
_output_shapes

:P*
dtype02&
$Hidden_layer_0/MatMul/ReadVariableOp�
Hidden_layer_0/MatMulMatMulinputs,Hidden_layer_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P2
Hidden_layer_0/MatMul�
%Hidden_layer_0/BiasAdd/ReadVariableOpReadVariableOp.hidden_layer_0_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype02'
%Hidden_layer_0/BiasAdd/ReadVariableOp�
Hidden_layer_0/BiasAddBiasAddHidden_layer_0/MatMul:product:0-Hidden_layer_0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P2
Hidden_layer_0/BiasAdd�
Hidden_layer_0/EluEluHidden_layer_0/BiasAdd:output:0*
T0*'
_output_shapes
:���������P2
Hidden_layer_0/Elu�
$Hidden_layer_1/MatMul/ReadVariableOpReadVariableOp-hidden_layer_1_matmul_readvariableop_resource*
_output_shapes

:PP*
dtype02&
$Hidden_layer_1/MatMul/ReadVariableOp�
Hidden_layer_1/MatMulMatMul Hidden_layer_0/Elu:activations:0,Hidden_layer_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P2
Hidden_layer_1/MatMul�
%Hidden_layer_1/BiasAdd/ReadVariableOpReadVariableOp.hidden_layer_1_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype02'
%Hidden_layer_1/BiasAdd/ReadVariableOp�
Hidden_layer_1/BiasAddBiasAddHidden_layer_1/MatMul:product:0-Hidden_layer_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P2
Hidden_layer_1/BiasAdd�
Hidden_layer_1/EluEluHidden_layer_1/BiasAdd:output:0*
T0*'
_output_shapes
:���������P2
Hidden_layer_1/Elu�
$Hidden_layer_2/MatMul/ReadVariableOpReadVariableOp-hidden_layer_2_matmul_readvariableop_resource*
_output_shapes

:PP*
dtype02&
$Hidden_layer_2/MatMul/ReadVariableOp�
Hidden_layer_2/MatMulMatMul Hidden_layer_1/Elu:activations:0,Hidden_layer_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P2
Hidden_layer_2/MatMul�
%Hidden_layer_2/BiasAdd/ReadVariableOpReadVariableOp.hidden_layer_2_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype02'
%Hidden_layer_2/BiasAdd/ReadVariableOp�
Hidden_layer_2/BiasAddBiasAddHidden_layer_2/MatMul:product:0-Hidden_layer_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P2
Hidden_layer_2/BiasAdd�
Hidden_layer_2/EluEluHidden_layer_2/BiasAdd:output:0*
T0*'
_output_shapes
:���������P2
Hidden_layer_2/Elu�
$Hidden_layer_3/MatMul/ReadVariableOpReadVariableOp-hidden_layer_3_matmul_readvariableop_resource*
_output_shapes

:PP*
dtype02&
$Hidden_layer_3/MatMul/ReadVariableOp�
Hidden_layer_3/MatMulMatMul Hidden_layer_2/Elu:activations:0,Hidden_layer_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P2
Hidden_layer_3/MatMul�
%Hidden_layer_3/BiasAdd/ReadVariableOpReadVariableOp.hidden_layer_3_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype02'
%Hidden_layer_3/BiasAdd/ReadVariableOp�
Hidden_layer_3/BiasAddBiasAddHidden_layer_3/MatMul:product:0-Hidden_layer_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P2
Hidden_layer_3/BiasAdd�
Hidden_layer_3/EluEluHidden_layer_3/BiasAdd:output:0*
T0*'
_output_shapes
:���������P2
Hidden_layer_3/Elu�
"Output_layer/MatMul/ReadVariableOpReadVariableOp+output_layer_matmul_readvariableop_resource*
_output_shapes

:P*
dtype02$
"Output_layer/MatMul/ReadVariableOp�
Output_layer/MatMulMatMul Hidden_layer_3/Elu:activations:0*Output_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
Output_layer/MatMul�
#Output_layer/BiasAdd/ReadVariableOpReadVariableOp,output_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02%
#Output_layer/BiasAdd/ReadVariableOp�
Output_layer/BiasAddBiasAddOutput_layer/MatMul:product:0+Output_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
Output_layer/BiasAdd|
Output_layer/EluEluOutput_layer/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
Output_layer/Elu�
7Hidden_layer_0/kernel/Regularizer/Square/ReadVariableOpReadVariableOp-hidden_layer_0_matmul_readvariableop_resource*
_output_shapes

:P*
dtype029
7Hidden_layer_0/kernel/Regularizer/Square/ReadVariableOp�
(Hidden_layer_0/kernel/Regularizer/SquareSquare?Hidden_layer_0/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:P2*
(Hidden_layer_0/kernel/Regularizer/Square�
'Hidden_layer_0/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2)
'Hidden_layer_0/kernel/Regularizer/Const�
%Hidden_layer_0/kernel/Regularizer/SumSum,Hidden_layer_0/kernel/Regularizer/Square:y:00Hidden_layer_0/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2'
%Hidden_layer_0/kernel/Regularizer/Sum�
'Hidden_layer_0/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2)
'Hidden_layer_0/kernel/Regularizer/mul/x�
%Hidden_layer_0/kernel/Regularizer/mulMul0Hidden_layer_0/kernel/Regularizer/mul/x:output:0.Hidden_layer_0/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2'
%Hidden_layer_0/kernel/Regularizer/mul�
7Hidden_layer_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp-hidden_layer_1_matmul_readvariableop_resource*
_output_shapes

:PP*
dtype029
7Hidden_layer_1/kernel/Regularizer/Square/ReadVariableOp�
(Hidden_layer_1/kernel/Regularizer/SquareSquare?Hidden_layer_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:PP2*
(Hidden_layer_1/kernel/Regularizer/Square�
'Hidden_layer_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2)
'Hidden_layer_1/kernel/Regularizer/Const�
%Hidden_layer_1/kernel/Regularizer/SumSum,Hidden_layer_1/kernel/Regularizer/Square:y:00Hidden_layer_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2'
%Hidden_layer_1/kernel/Regularizer/Sum�
'Hidden_layer_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2)
'Hidden_layer_1/kernel/Regularizer/mul/x�
%Hidden_layer_1/kernel/Regularizer/mulMul0Hidden_layer_1/kernel/Regularizer/mul/x:output:0.Hidden_layer_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2'
%Hidden_layer_1/kernel/Regularizer/mul�
7Hidden_layer_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp-hidden_layer_2_matmul_readvariableop_resource*
_output_shapes

:PP*
dtype029
7Hidden_layer_2/kernel/Regularizer/Square/ReadVariableOp�
(Hidden_layer_2/kernel/Regularizer/SquareSquare?Hidden_layer_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:PP2*
(Hidden_layer_2/kernel/Regularizer/Square�
'Hidden_layer_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2)
'Hidden_layer_2/kernel/Regularizer/Const�
%Hidden_layer_2/kernel/Regularizer/SumSum,Hidden_layer_2/kernel/Regularizer/Square:y:00Hidden_layer_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2'
%Hidden_layer_2/kernel/Regularizer/Sum�
'Hidden_layer_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2)
'Hidden_layer_2/kernel/Regularizer/mul/x�
%Hidden_layer_2/kernel/Regularizer/mulMul0Hidden_layer_2/kernel/Regularizer/mul/x:output:0.Hidden_layer_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2'
%Hidden_layer_2/kernel/Regularizer/mul�
7Hidden_layer_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOp-hidden_layer_3_matmul_readvariableop_resource*
_output_shapes

:PP*
dtype029
7Hidden_layer_3/kernel/Regularizer/Square/ReadVariableOp�
(Hidden_layer_3/kernel/Regularizer/SquareSquare?Hidden_layer_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:PP2*
(Hidden_layer_3/kernel/Regularizer/Square�
'Hidden_layer_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2)
'Hidden_layer_3/kernel/Regularizer/Const�
%Hidden_layer_3/kernel/Regularizer/SumSum,Hidden_layer_3/kernel/Regularizer/Square:y:00Hidden_layer_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2'
%Hidden_layer_3/kernel/Regularizer/Sum�
'Hidden_layer_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2)
'Hidden_layer_3/kernel/Regularizer/mul/x�
%Hidden_layer_3/kernel/Regularizer/mulMul0Hidden_layer_3/kernel/Regularizer/mul/x:output:0.Hidden_layer_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2'
%Hidden_layer_3/kernel/Regularizer/mul�
IdentityIdentityOutput_layer/Elu:activations:0&^Hidden_layer_0/BiasAdd/ReadVariableOp%^Hidden_layer_0/MatMul/ReadVariableOp8^Hidden_layer_0/kernel/Regularizer/Square/ReadVariableOp&^Hidden_layer_1/BiasAdd/ReadVariableOp%^Hidden_layer_1/MatMul/ReadVariableOp8^Hidden_layer_1/kernel/Regularizer/Square/ReadVariableOp&^Hidden_layer_2/BiasAdd/ReadVariableOp%^Hidden_layer_2/MatMul/ReadVariableOp8^Hidden_layer_2/kernel/Regularizer/Square/ReadVariableOp&^Hidden_layer_3/BiasAdd/ReadVariableOp%^Hidden_layer_3/MatMul/ReadVariableOp8^Hidden_layer_3/kernel/Regularizer/Square/ReadVariableOp$^Output_layer/BiasAdd/ReadVariableOp#^Output_layer/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:���������::::::::::2N
%Hidden_layer_0/BiasAdd/ReadVariableOp%Hidden_layer_0/BiasAdd/ReadVariableOp2L
$Hidden_layer_0/MatMul/ReadVariableOp$Hidden_layer_0/MatMul/ReadVariableOp2r
7Hidden_layer_0/kernel/Regularizer/Square/ReadVariableOp7Hidden_layer_0/kernel/Regularizer/Square/ReadVariableOp2N
%Hidden_layer_1/BiasAdd/ReadVariableOp%Hidden_layer_1/BiasAdd/ReadVariableOp2L
$Hidden_layer_1/MatMul/ReadVariableOp$Hidden_layer_1/MatMul/ReadVariableOp2r
7Hidden_layer_1/kernel/Regularizer/Square/ReadVariableOp7Hidden_layer_1/kernel/Regularizer/Square/ReadVariableOp2N
%Hidden_layer_2/BiasAdd/ReadVariableOp%Hidden_layer_2/BiasAdd/ReadVariableOp2L
$Hidden_layer_2/MatMul/ReadVariableOp$Hidden_layer_2/MatMul/ReadVariableOp2r
7Hidden_layer_2/kernel/Regularizer/Square/ReadVariableOp7Hidden_layer_2/kernel/Regularizer/Square/ReadVariableOp2N
%Hidden_layer_3/BiasAdd/ReadVariableOp%Hidden_layer_3/BiasAdd/ReadVariableOp2L
$Hidden_layer_3/MatMul/ReadVariableOp$Hidden_layer_3/MatMul/ReadVariableOp2r
7Hidden_layer_3/kernel/Regularizer/Square/ReadVariableOp7Hidden_layer_3/kernel/Regularizer/Square/ReadVariableOp2J
#Output_layer/BiasAdd/ReadVariableOp#Output_layer/BiasAdd/ReadVariableOp2H
"Output_layer/MatMul/ReadVariableOp"Output_layer/MatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�A
�	
!__inference__wrapped_model_243294
input_layer<
8sequential_hidden_layer_0_matmul_readvariableop_resource=
9sequential_hidden_layer_0_biasadd_readvariableop_resource<
8sequential_hidden_layer_1_matmul_readvariableop_resource=
9sequential_hidden_layer_1_biasadd_readvariableop_resource<
8sequential_hidden_layer_2_matmul_readvariableop_resource=
9sequential_hidden_layer_2_biasadd_readvariableop_resource<
8sequential_hidden_layer_3_matmul_readvariableop_resource=
9sequential_hidden_layer_3_biasadd_readvariableop_resource:
6sequential_output_layer_matmul_readvariableop_resource;
7sequential_output_layer_biasadd_readvariableop_resource
identity��0sequential/Hidden_layer_0/BiasAdd/ReadVariableOp�/sequential/Hidden_layer_0/MatMul/ReadVariableOp�0sequential/Hidden_layer_1/BiasAdd/ReadVariableOp�/sequential/Hidden_layer_1/MatMul/ReadVariableOp�0sequential/Hidden_layer_2/BiasAdd/ReadVariableOp�/sequential/Hidden_layer_2/MatMul/ReadVariableOp�0sequential/Hidden_layer_3/BiasAdd/ReadVariableOp�/sequential/Hidden_layer_3/MatMul/ReadVariableOp�.sequential/Output_layer/BiasAdd/ReadVariableOp�-sequential/Output_layer/MatMul/ReadVariableOp�
/sequential/Hidden_layer_0/MatMul/ReadVariableOpReadVariableOp8sequential_hidden_layer_0_matmul_readvariableop_resource*
_output_shapes

:P*
dtype021
/sequential/Hidden_layer_0/MatMul/ReadVariableOp�
 sequential/Hidden_layer_0/MatMulMatMulinput_layer7sequential/Hidden_layer_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P2"
 sequential/Hidden_layer_0/MatMul�
0sequential/Hidden_layer_0/BiasAdd/ReadVariableOpReadVariableOp9sequential_hidden_layer_0_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype022
0sequential/Hidden_layer_0/BiasAdd/ReadVariableOp�
!sequential/Hidden_layer_0/BiasAddBiasAdd*sequential/Hidden_layer_0/MatMul:product:08sequential/Hidden_layer_0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P2#
!sequential/Hidden_layer_0/BiasAdd�
sequential/Hidden_layer_0/EluElu*sequential/Hidden_layer_0/BiasAdd:output:0*
T0*'
_output_shapes
:���������P2
sequential/Hidden_layer_0/Elu�
/sequential/Hidden_layer_1/MatMul/ReadVariableOpReadVariableOp8sequential_hidden_layer_1_matmul_readvariableop_resource*
_output_shapes

:PP*
dtype021
/sequential/Hidden_layer_1/MatMul/ReadVariableOp�
 sequential/Hidden_layer_1/MatMulMatMul+sequential/Hidden_layer_0/Elu:activations:07sequential/Hidden_layer_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P2"
 sequential/Hidden_layer_1/MatMul�
0sequential/Hidden_layer_1/BiasAdd/ReadVariableOpReadVariableOp9sequential_hidden_layer_1_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype022
0sequential/Hidden_layer_1/BiasAdd/ReadVariableOp�
!sequential/Hidden_layer_1/BiasAddBiasAdd*sequential/Hidden_layer_1/MatMul:product:08sequential/Hidden_layer_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P2#
!sequential/Hidden_layer_1/BiasAdd�
sequential/Hidden_layer_1/EluElu*sequential/Hidden_layer_1/BiasAdd:output:0*
T0*'
_output_shapes
:���������P2
sequential/Hidden_layer_1/Elu�
/sequential/Hidden_layer_2/MatMul/ReadVariableOpReadVariableOp8sequential_hidden_layer_2_matmul_readvariableop_resource*
_output_shapes

:PP*
dtype021
/sequential/Hidden_layer_2/MatMul/ReadVariableOp�
 sequential/Hidden_layer_2/MatMulMatMul+sequential/Hidden_layer_1/Elu:activations:07sequential/Hidden_layer_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P2"
 sequential/Hidden_layer_2/MatMul�
0sequential/Hidden_layer_2/BiasAdd/ReadVariableOpReadVariableOp9sequential_hidden_layer_2_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype022
0sequential/Hidden_layer_2/BiasAdd/ReadVariableOp�
!sequential/Hidden_layer_2/BiasAddBiasAdd*sequential/Hidden_layer_2/MatMul:product:08sequential/Hidden_layer_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P2#
!sequential/Hidden_layer_2/BiasAdd�
sequential/Hidden_layer_2/EluElu*sequential/Hidden_layer_2/BiasAdd:output:0*
T0*'
_output_shapes
:���������P2
sequential/Hidden_layer_2/Elu�
/sequential/Hidden_layer_3/MatMul/ReadVariableOpReadVariableOp8sequential_hidden_layer_3_matmul_readvariableop_resource*
_output_shapes

:PP*
dtype021
/sequential/Hidden_layer_3/MatMul/ReadVariableOp�
 sequential/Hidden_layer_3/MatMulMatMul+sequential/Hidden_layer_2/Elu:activations:07sequential/Hidden_layer_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P2"
 sequential/Hidden_layer_3/MatMul�
0sequential/Hidden_layer_3/BiasAdd/ReadVariableOpReadVariableOp9sequential_hidden_layer_3_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype022
0sequential/Hidden_layer_3/BiasAdd/ReadVariableOp�
!sequential/Hidden_layer_3/BiasAddBiasAdd*sequential/Hidden_layer_3/MatMul:product:08sequential/Hidden_layer_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P2#
!sequential/Hidden_layer_3/BiasAdd�
sequential/Hidden_layer_3/EluElu*sequential/Hidden_layer_3/BiasAdd:output:0*
T0*'
_output_shapes
:���������P2
sequential/Hidden_layer_3/Elu�
-sequential/Output_layer/MatMul/ReadVariableOpReadVariableOp6sequential_output_layer_matmul_readvariableop_resource*
_output_shapes

:P*
dtype02/
-sequential/Output_layer/MatMul/ReadVariableOp�
sequential/Output_layer/MatMulMatMul+sequential/Hidden_layer_3/Elu:activations:05sequential/Output_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2 
sequential/Output_layer/MatMul�
.sequential/Output_layer/BiasAdd/ReadVariableOpReadVariableOp7sequential_output_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype020
.sequential/Output_layer/BiasAdd/ReadVariableOp�
sequential/Output_layer/BiasAddBiasAdd(sequential/Output_layer/MatMul:product:06sequential/Output_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2!
sequential/Output_layer/BiasAdd�
sequential/Output_layer/EluElu(sequential/Output_layer/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
sequential/Output_layer/Elu�
IdentityIdentity)sequential/Output_layer/Elu:activations:01^sequential/Hidden_layer_0/BiasAdd/ReadVariableOp0^sequential/Hidden_layer_0/MatMul/ReadVariableOp1^sequential/Hidden_layer_1/BiasAdd/ReadVariableOp0^sequential/Hidden_layer_1/MatMul/ReadVariableOp1^sequential/Hidden_layer_2/BiasAdd/ReadVariableOp0^sequential/Hidden_layer_2/MatMul/ReadVariableOp1^sequential/Hidden_layer_3/BiasAdd/ReadVariableOp0^sequential/Hidden_layer_3/MatMul/ReadVariableOp/^sequential/Output_layer/BiasAdd/ReadVariableOp.^sequential/Output_layer/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:���������::::::::::2d
0sequential/Hidden_layer_0/BiasAdd/ReadVariableOp0sequential/Hidden_layer_0/BiasAdd/ReadVariableOp2b
/sequential/Hidden_layer_0/MatMul/ReadVariableOp/sequential/Hidden_layer_0/MatMul/ReadVariableOp2d
0sequential/Hidden_layer_1/BiasAdd/ReadVariableOp0sequential/Hidden_layer_1/BiasAdd/ReadVariableOp2b
/sequential/Hidden_layer_1/MatMul/ReadVariableOp/sequential/Hidden_layer_1/MatMul/ReadVariableOp2d
0sequential/Hidden_layer_2/BiasAdd/ReadVariableOp0sequential/Hidden_layer_2/BiasAdd/ReadVariableOp2b
/sequential/Hidden_layer_2/MatMul/ReadVariableOp/sequential/Hidden_layer_2/MatMul/ReadVariableOp2d
0sequential/Hidden_layer_3/BiasAdd/ReadVariableOp0sequential/Hidden_layer_3/BiasAdd/ReadVariableOp2b
/sequential/Hidden_layer_3/MatMul/ReadVariableOp/sequential/Hidden_layer_3/MatMul/ReadVariableOp2`
.sequential/Output_layer/BiasAdd/ReadVariableOp.sequential/Output_layer/BiasAdd/ReadVariableOp2^
-sequential/Output_layer/MatMul/ReadVariableOp-sequential/Output_layer/MatMul/ReadVariableOp:T P
'
_output_shapes
:���������
%
_user_specified_nameInput_layer
�
�
/__inference_Hidden_layer_0_layer_call_fn_243959

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������P*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_Hidden_layer_0_layer_call_and_return_conditional_losses_2433152
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������P2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
__inference_loss_fn_2_244108D
@hidden_layer_2_kernel_regularizer_square_readvariableop_resource
identity��7Hidden_layer_2/kernel/Regularizer/Square/ReadVariableOp�
7Hidden_layer_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp@hidden_layer_2_kernel_regularizer_square_readvariableop_resource*
_output_shapes

:PP*
dtype029
7Hidden_layer_2/kernel/Regularizer/Square/ReadVariableOp�
(Hidden_layer_2/kernel/Regularizer/SquareSquare?Hidden_layer_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:PP2*
(Hidden_layer_2/kernel/Regularizer/Square�
'Hidden_layer_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2)
'Hidden_layer_2/kernel/Regularizer/Const�
%Hidden_layer_2/kernel/Regularizer/SumSum,Hidden_layer_2/kernel/Regularizer/Square:y:00Hidden_layer_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2'
%Hidden_layer_2/kernel/Regularizer/Sum�
'Hidden_layer_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2)
'Hidden_layer_2/kernel/Regularizer/mul/x�
%Hidden_layer_2/kernel/Regularizer/mulMul0Hidden_layer_2/kernel/Regularizer/mul/x:output:0.Hidden_layer_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2'
%Hidden_layer_2/kernel/Regularizer/mul�
IdentityIdentity)Hidden_layer_2/kernel/Regularizer/mul:z:08^Hidden_layer_2/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:2r
7Hidden_layer_2/kernel/Regularizer/Square/ReadVariableOp7Hidden_layer_2/kernel/Regularizer/Square/ReadVariableOp
�I
�
F__inference_sequential_layer_call_and_return_conditional_losses_243535
input_layer
hidden_layer_0_243485
hidden_layer_0_243487
hidden_layer_1_243490
hidden_layer_1_243492
hidden_layer_2_243495
hidden_layer_2_243497
hidden_layer_3_243500
hidden_layer_3_243502
output_layer_243505
output_layer_243507
identity��&Hidden_layer_0/StatefulPartitionedCall�7Hidden_layer_0/kernel/Regularizer/Square/ReadVariableOp�&Hidden_layer_1/StatefulPartitionedCall�7Hidden_layer_1/kernel/Regularizer/Square/ReadVariableOp�&Hidden_layer_2/StatefulPartitionedCall�7Hidden_layer_2/kernel/Regularizer/Square/ReadVariableOp�&Hidden_layer_3/StatefulPartitionedCall�7Hidden_layer_3/kernel/Regularizer/Square/ReadVariableOp�$Output_layer/StatefulPartitionedCall�
&Hidden_layer_0/StatefulPartitionedCallStatefulPartitionedCallinput_layerhidden_layer_0_243485hidden_layer_0_243487*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������P*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_Hidden_layer_0_layer_call_and_return_conditional_losses_2433152(
&Hidden_layer_0/StatefulPartitionedCall�
&Hidden_layer_1/StatefulPartitionedCallStatefulPartitionedCall/Hidden_layer_0/StatefulPartitionedCall:output:0hidden_layer_1_243490hidden_layer_1_243492*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������P*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_Hidden_layer_1_layer_call_and_return_conditional_losses_2433482(
&Hidden_layer_1/StatefulPartitionedCall�
&Hidden_layer_2/StatefulPartitionedCallStatefulPartitionedCall/Hidden_layer_1/StatefulPartitionedCall:output:0hidden_layer_2_243495hidden_layer_2_243497*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������P*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_Hidden_layer_2_layer_call_and_return_conditional_losses_2433812(
&Hidden_layer_2/StatefulPartitionedCall�
&Hidden_layer_3/StatefulPartitionedCallStatefulPartitionedCall/Hidden_layer_2/StatefulPartitionedCall:output:0hidden_layer_3_243500hidden_layer_3_243502*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������P*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_Hidden_layer_3_layer_call_and_return_conditional_losses_2434142(
&Hidden_layer_3/StatefulPartitionedCall�
$Output_layer/StatefulPartitionedCallStatefulPartitionedCall/Hidden_layer_3/StatefulPartitionedCall:output:0output_layer_243505output_layer_243507*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_Output_layer_layer_call_and_return_conditional_losses_2434412&
$Output_layer/StatefulPartitionedCall�
7Hidden_layer_0/kernel/Regularizer/Square/ReadVariableOpReadVariableOphidden_layer_0_243485*
_output_shapes

:P*
dtype029
7Hidden_layer_0/kernel/Regularizer/Square/ReadVariableOp�
(Hidden_layer_0/kernel/Regularizer/SquareSquare?Hidden_layer_0/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:P2*
(Hidden_layer_0/kernel/Regularizer/Square�
'Hidden_layer_0/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2)
'Hidden_layer_0/kernel/Regularizer/Const�
%Hidden_layer_0/kernel/Regularizer/SumSum,Hidden_layer_0/kernel/Regularizer/Square:y:00Hidden_layer_0/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2'
%Hidden_layer_0/kernel/Regularizer/Sum�
'Hidden_layer_0/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2)
'Hidden_layer_0/kernel/Regularizer/mul/x�
%Hidden_layer_0/kernel/Regularizer/mulMul0Hidden_layer_0/kernel/Regularizer/mul/x:output:0.Hidden_layer_0/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2'
%Hidden_layer_0/kernel/Regularizer/mul�
7Hidden_layer_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOphidden_layer_1_243490*
_output_shapes

:PP*
dtype029
7Hidden_layer_1/kernel/Regularizer/Square/ReadVariableOp�
(Hidden_layer_1/kernel/Regularizer/SquareSquare?Hidden_layer_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:PP2*
(Hidden_layer_1/kernel/Regularizer/Square�
'Hidden_layer_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2)
'Hidden_layer_1/kernel/Regularizer/Const�
%Hidden_layer_1/kernel/Regularizer/SumSum,Hidden_layer_1/kernel/Regularizer/Square:y:00Hidden_layer_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2'
%Hidden_layer_1/kernel/Regularizer/Sum�
'Hidden_layer_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2)
'Hidden_layer_1/kernel/Regularizer/mul/x�
%Hidden_layer_1/kernel/Regularizer/mulMul0Hidden_layer_1/kernel/Regularizer/mul/x:output:0.Hidden_layer_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2'
%Hidden_layer_1/kernel/Regularizer/mul�
7Hidden_layer_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOphidden_layer_2_243495*
_output_shapes

:PP*
dtype029
7Hidden_layer_2/kernel/Regularizer/Square/ReadVariableOp�
(Hidden_layer_2/kernel/Regularizer/SquareSquare?Hidden_layer_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:PP2*
(Hidden_layer_2/kernel/Regularizer/Square�
'Hidden_layer_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2)
'Hidden_layer_2/kernel/Regularizer/Const�
%Hidden_layer_2/kernel/Regularizer/SumSum,Hidden_layer_2/kernel/Regularizer/Square:y:00Hidden_layer_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2'
%Hidden_layer_2/kernel/Regularizer/Sum�
'Hidden_layer_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2)
'Hidden_layer_2/kernel/Regularizer/mul/x�
%Hidden_layer_2/kernel/Regularizer/mulMul0Hidden_layer_2/kernel/Regularizer/mul/x:output:0.Hidden_layer_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2'
%Hidden_layer_2/kernel/Regularizer/mul�
7Hidden_layer_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOphidden_layer_3_243500*
_output_shapes

:PP*
dtype029
7Hidden_layer_3/kernel/Regularizer/Square/ReadVariableOp�
(Hidden_layer_3/kernel/Regularizer/SquareSquare?Hidden_layer_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:PP2*
(Hidden_layer_3/kernel/Regularizer/Square�
'Hidden_layer_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2)
'Hidden_layer_3/kernel/Regularizer/Const�
%Hidden_layer_3/kernel/Regularizer/SumSum,Hidden_layer_3/kernel/Regularizer/Square:y:00Hidden_layer_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2'
%Hidden_layer_3/kernel/Regularizer/Sum�
'Hidden_layer_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2)
'Hidden_layer_3/kernel/Regularizer/mul/x�
%Hidden_layer_3/kernel/Regularizer/mulMul0Hidden_layer_3/kernel/Regularizer/mul/x:output:0.Hidden_layer_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2'
%Hidden_layer_3/kernel/Regularizer/mul�
IdentityIdentity-Output_layer/StatefulPartitionedCall:output:0'^Hidden_layer_0/StatefulPartitionedCall8^Hidden_layer_0/kernel/Regularizer/Square/ReadVariableOp'^Hidden_layer_1/StatefulPartitionedCall8^Hidden_layer_1/kernel/Regularizer/Square/ReadVariableOp'^Hidden_layer_2/StatefulPartitionedCall8^Hidden_layer_2/kernel/Regularizer/Square/ReadVariableOp'^Hidden_layer_3/StatefulPartitionedCall8^Hidden_layer_3/kernel/Regularizer/Square/ReadVariableOp%^Output_layer/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:���������::::::::::2P
&Hidden_layer_0/StatefulPartitionedCall&Hidden_layer_0/StatefulPartitionedCall2r
7Hidden_layer_0/kernel/Regularizer/Square/ReadVariableOp7Hidden_layer_0/kernel/Regularizer/Square/ReadVariableOp2P
&Hidden_layer_1/StatefulPartitionedCall&Hidden_layer_1/StatefulPartitionedCall2r
7Hidden_layer_1/kernel/Regularizer/Square/ReadVariableOp7Hidden_layer_1/kernel/Regularizer/Square/ReadVariableOp2P
&Hidden_layer_2/StatefulPartitionedCall&Hidden_layer_2/StatefulPartitionedCall2r
7Hidden_layer_2/kernel/Regularizer/Square/ReadVariableOp7Hidden_layer_2/kernel/Regularizer/Square/ReadVariableOp2P
&Hidden_layer_3/StatefulPartitionedCall&Hidden_layer_3/StatefulPartitionedCall2r
7Hidden_layer_3/kernel/Regularizer/Square/ReadVariableOp7Hidden_layer_3/kernel/Regularizer/Square/ReadVariableOp2L
$Output_layer/StatefulPartitionedCall$Output_layer/StatefulPartitionedCall:T P
'
_output_shapes
:���������
%
_user_specified_nameInput_layer
�I
�
F__inference_sequential_layer_call_and_return_conditional_losses_243669

inputs
hidden_layer_0_243619
hidden_layer_0_243621
hidden_layer_1_243624
hidden_layer_1_243626
hidden_layer_2_243629
hidden_layer_2_243631
hidden_layer_3_243634
hidden_layer_3_243636
output_layer_243639
output_layer_243641
identity��&Hidden_layer_0/StatefulPartitionedCall�7Hidden_layer_0/kernel/Regularizer/Square/ReadVariableOp�&Hidden_layer_1/StatefulPartitionedCall�7Hidden_layer_1/kernel/Regularizer/Square/ReadVariableOp�&Hidden_layer_2/StatefulPartitionedCall�7Hidden_layer_2/kernel/Regularizer/Square/ReadVariableOp�&Hidden_layer_3/StatefulPartitionedCall�7Hidden_layer_3/kernel/Regularizer/Square/ReadVariableOp�$Output_layer/StatefulPartitionedCall�
&Hidden_layer_0/StatefulPartitionedCallStatefulPartitionedCallinputshidden_layer_0_243619hidden_layer_0_243621*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������P*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_Hidden_layer_0_layer_call_and_return_conditional_losses_2433152(
&Hidden_layer_0/StatefulPartitionedCall�
&Hidden_layer_1/StatefulPartitionedCallStatefulPartitionedCall/Hidden_layer_0/StatefulPartitionedCall:output:0hidden_layer_1_243624hidden_layer_1_243626*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������P*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_Hidden_layer_1_layer_call_and_return_conditional_losses_2433482(
&Hidden_layer_1/StatefulPartitionedCall�
&Hidden_layer_2/StatefulPartitionedCallStatefulPartitionedCall/Hidden_layer_1/StatefulPartitionedCall:output:0hidden_layer_2_243629hidden_layer_2_243631*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������P*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_Hidden_layer_2_layer_call_and_return_conditional_losses_2433812(
&Hidden_layer_2/StatefulPartitionedCall�
&Hidden_layer_3/StatefulPartitionedCallStatefulPartitionedCall/Hidden_layer_2/StatefulPartitionedCall:output:0hidden_layer_3_243634hidden_layer_3_243636*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������P*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_Hidden_layer_3_layer_call_and_return_conditional_losses_2434142(
&Hidden_layer_3/StatefulPartitionedCall�
$Output_layer/StatefulPartitionedCallStatefulPartitionedCall/Hidden_layer_3/StatefulPartitionedCall:output:0output_layer_243639output_layer_243641*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_Output_layer_layer_call_and_return_conditional_losses_2434412&
$Output_layer/StatefulPartitionedCall�
7Hidden_layer_0/kernel/Regularizer/Square/ReadVariableOpReadVariableOphidden_layer_0_243619*
_output_shapes

:P*
dtype029
7Hidden_layer_0/kernel/Regularizer/Square/ReadVariableOp�
(Hidden_layer_0/kernel/Regularizer/SquareSquare?Hidden_layer_0/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:P2*
(Hidden_layer_0/kernel/Regularizer/Square�
'Hidden_layer_0/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2)
'Hidden_layer_0/kernel/Regularizer/Const�
%Hidden_layer_0/kernel/Regularizer/SumSum,Hidden_layer_0/kernel/Regularizer/Square:y:00Hidden_layer_0/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2'
%Hidden_layer_0/kernel/Regularizer/Sum�
'Hidden_layer_0/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2)
'Hidden_layer_0/kernel/Regularizer/mul/x�
%Hidden_layer_0/kernel/Regularizer/mulMul0Hidden_layer_0/kernel/Regularizer/mul/x:output:0.Hidden_layer_0/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2'
%Hidden_layer_0/kernel/Regularizer/mul�
7Hidden_layer_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOphidden_layer_1_243624*
_output_shapes

:PP*
dtype029
7Hidden_layer_1/kernel/Regularizer/Square/ReadVariableOp�
(Hidden_layer_1/kernel/Regularizer/SquareSquare?Hidden_layer_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:PP2*
(Hidden_layer_1/kernel/Regularizer/Square�
'Hidden_layer_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2)
'Hidden_layer_1/kernel/Regularizer/Const�
%Hidden_layer_1/kernel/Regularizer/SumSum,Hidden_layer_1/kernel/Regularizer/Square:y:00Hidden_layer_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2'
%Hidden_layer_1/kernel/Regularizer/Sum�
'Hidden_layer_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2)
'Hidden_layer_1/kernel/Regularizer/mul/x�
%Hidden_layer_1/kernel/Regularizer/mulMul0Hidden_layer_1/kernel/Regularizer/mul/x:output:0.Hidden_layer_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2'
%Hidden_layer_1/kernel/Regularizer/mul�
7Hidden_layer_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOphidden_layer_2_243629*
_output_shapes

:PP*
dtype029
7Hidden_layer_2/kernel/Regularizer/Square/ReadVariableOp�
(Hidden_layer_2/kernel/Regularizer/SquareSquare?Hidden_layer_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:PP2*
(Hidden_layer_2/kernel/Regularizer/Square�
'Hidden_layer_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2)
'Hidden_layer_2/kernel/Regularizer/Const�
%Hidden_layer_2/kernel/Regularizer/SumSum,Hidden_layer_2/kernel/Regularizer/Square:y:00Hidden_layer_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2'
%Hidden_layer_2/kernel/Regularizer/Sum�
'Hidden_layer_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2)
'Hidden_layer_2/kernel/Regularizer/mul/x�
%Hidden_layer_2/kernel/Regularizer/mulMul0Hidden_layer_2/kernel/Regularizer/mul/x:output:0.Hidden_layer_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2'
%Hidden_layer_2/kernel/Regularizer/mul�
7Hidden_layer_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOphidden_layer_3_243634*
_output_shapes

:PP*
dtype029
7Hidden_layer_3/kernel/Regularizer/Square/ReadVariableOp�
(Hidden_layer_3/kernel/Regularizer/SquareSquare?Hidden_layer_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:PP2*
(Hidden_layer_3/kernel/Regularizer/Square�
'Hidden_layer_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2)
'Hidden_layer_3/kernel/Regularizer/Const�
%Hidden_layer_3/kernel/Regularizer/SumSum,Hidden_layer_3/kernel/Regularizer/Square:y:00Hidden_layer_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2'
%Hidden_layer_3/kernel/Regularizer/Sum�
'Hidden_layer_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2)
'Hidden_layer_3/kernel/Regularizer/mul/x�
%Hidden_layer_3/kernel/Regularizer/mulMul0Hidden_layer_3/kernel/Regularizer/mul/x:output:0.Hidden_layer_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2'
%Hidden_layer_3/kernel/Regularizer/mul�
IdentityIdentity-Output_layer/StatefulPartitionedCall:output:0'^Hidden_layer_0/StatefulPartitionedCall8^Hidden_layer_0/kernel/Regularizer/Square/ReadVariableOp'^Hidden_layer_1/StatefulPartitionedCall8^Hidden_layer_1/kernel/Regularizer/Square/ReadVariableOp'^Hidden_layer_2/StatefulPartitionedCall8^Hidden_layer_2/kernel/Regularizer/Square/ReadVariableOp'^Hidden_layer_3/StatefulPartitionedCall8^Hidden_layer_3/kernel/Regularizer/Square/ReadVariableOp%^Output_layer/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:���������::::::::::2P
&Hidden_layer_0/StatefulPartitionedCall&Hidden_layer_0/StatefulPartitionedCall2r
7Hidden_layer_0/kernel/Regularizer/Square/ReadVariableOp7Hidden_layer_0/kernel/Regularizer/Square/ReadVariableOp2P
&Hidden_layer_1/StatefulPartitionedCall&Hidden_layer_1/StatefulPartitionedCall2r
7Hidden_layer_1/kernel/Regularizer/Square/ReadVariableOp7Hidden_layer_1/kernel/Regularizer/Square/ReadVariableOp2P
&Hidden_layer_2/StatefulPartitionedCall&Hidden_layer_2/StatefulPartitionedCall2r
7Hidden_layer_2/kernel/Regularizer/Square/ReadVariableOp7Hidden_layer_2/kernel/Regularizer/Square/ReadVariableOp2P
&Hidden_layer_3/StatefulPartitionedCall&Hidden_layer_3/StatefulPartitionedCall2r
7Hidden_layer_3/kernel/Regularizer/Square/ReadVariableOp7Hidden_layer_3/kernel/Regularizer/Square/ReadVariableOp2L
$Output_layer/StatefulPartitionedCall$Output_layer/StatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
J__inference_Hidden_layer_2_layer_call_and_return_conditional_losses_243381

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�7Hidden_layer_2/kernel/Regularizer/Square/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:PP*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:P*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P2	
BiasAddU
EluEluBiasAdd:output:0*
T0*'
_output_shapes
:���������P2
Elu�
7Hidden_layer_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:PP*
dtype029
7Hidden_layer_2/kernel/Regularizer/Square/ReadVariableOp�
(Hidden_layer_2/kernel/Regularizer/SquareSquare?Hidden_layer_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:PP2*
(Hidden_layer_2/kernel/Regularizer/Square�
'Hidden_layer_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2)
'Hidden_layer_2/kernel/Regularizer/Const�
%Hidden_layer_2/kernel/Regularizer/SumSum,Hidden_layer_2/kernel/Regularizer/Square:y:00Hidden_layer_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2'
%Hidden_layer_2/kernel/Regularizer/Sum�
'Hidden_layer_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2)
'Hidden_layer_2/kernel/Regularizer/mul/x�
%Hidden_layer_2/kernel/Regularizer/mulMul0Hidden_layer_2/kernel/Regularizer/mul/x:output:0.Hidden_layer_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2'
%Hidden_layer_2/kernel/Regularizer/mul�
IdentityIdentityElu:activations:0^BiasAdd/ReadVariableOp8^Hidden_layer_2/kernel/Regularizer/Square/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������P2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������P::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2r
7Hidden_layer_2/kernel/Regularizer/Square/ReadVariableOp7Hidden_layer_2/kernel/Regularizer/Square/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������P
 
_user_specified_nameinputs
�
�
+__inference_sequential_layer_call_fn_243614
input_layer
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_layerunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_2435912
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:���������::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
'
_output_shapes
:���������
%
_user_specified_nameInput_layer
�	
�
H__inference_Output_layer_layer_call_and_return_conditional_losses_244066

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:P*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAddU
EluEluBiasAdd:output:0*
T0*'
_output_shapes
:���������2
Elu�
IdentityIdentityElu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������P::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������P
 
_user_specified_nameinputs
�
�
/__inference_Hidden_layer_2_layer_call_fn_244023

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������P*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_Hidden_layer_2_layer_call_and_return_conditional_losses_2433812
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������P2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������P::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������P
 
_user_specified_nameinputs
�
�
__inference_loss_fn_1_244097D
@hidden_layer_1_kernel_regularizer_square_readvariableop_resource
identity��7Hidden_layer_1/kernel/Regularizer/Square/ReadVariableOp�
7Hidden_layer_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp@hidden_layer_1_kernel_regularizer_square_readvariableop_resource*
_output_shapes

:PP*
dtype029
7Hidden_layer_1/kernel/Regularizer/Square/ReadVariableOp�
(Hidden_layer_1/kernel/Regularizer/SquareSquare?Hidden_layer_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:PP2*
(Hidden_layer_1/kernel/Regularizer/Square�
'Hidden_layer_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2)
'Hidden_layer_1/kernel/Regularizer/Const�
%Hidden_layer_1/kernel/Regularizer/SumSum,Hidden_layer_1/kernel/Regularizer/Square:y:00Hidden_layer_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2'
%Hidden_layer_1/kernel/Regularizer/Sum�
'Hidden_layer_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2)
'Hidden_layer_1/kernel/Regularizer/mul/x�
%Hidden_layer_1/kernel/Regularizer/mulMul0Hidden_layer_1/kernel/Regularizer/mul/x:output:0.Hidden_layer_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2'
%Hidden_layer_1/kernel/Regularizer/mul�
IdentityIdentity)Hidden_layer_1/kernel/Regularizer/mul:z:08^Hidden_layer_1/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:2r
7Hidden_layer_1/kernel/Regularizer/Square/ReadVariableOp7Hidden_layer_1/kernel/Regularizer/Square/ReadVariableOp
�
�
J__inference_Hidden_layer_0_layer_call_and_return_conditional_losses_243950

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�7Hidden_layer_0/kernel/Regularizer/Square/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:P*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:P*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P2	
BiasAddU
EluEluBiasAdd:output:0*
T0*'
_output_shapes
:���������P2
Elu�
7Hidden_layer_0/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:P*
dtype029
7Hidden_layer_0/kernel/Regularizer/Square/ReadVariableOp�
(Hidden_layer_0/kernel/Regularizer/SquareSquare?Hidden_layer_0/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:P2*
(Hidden_layer_0/kernel/Regularizer/Square�
'Hidden_layer_0/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2)
'Hidden_layer_0/kernel/Regularizer/Const�
%Hidden_layer_0/kernel/Regularizer/SumSum,Hidden_layer_0/kernel/Regularizer/Square:y:00Hidden_layer_0/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2'
%Hidden_layer_0/kernel/Regularizer/Sum�
'Hidden_layer_0/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2)
'Hidden_layer_0/kernel/Regularizer/mul/x�
%Hidden_layer_0/kernel/Regularizer/mulMul0Hidden_layer_0/kernel/Regularizer/mul/x:output:0.Hidden_layer_0/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2'
%Hidden_layer_0/kernel/Regularizer/mul�
IdentityIdentityElu:activations:0^BiasAdd/ReadVariableOp8^Hidden_layer_0/kernel/Regularizer/Square/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������P2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2r
7Hidden_layer_0/kernel/Regularizer/Square/ReadVariableOp7Hidden_layer_0/kernel/Regularizer/Square/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
/__inference_Hidden_layer_1_layer_call_fn_243991

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������P*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_Hidden_layer_1_layer_call_and_return_conditional_losses_2433482
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������P2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������P::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������P
 
_user_specified_nameinputs
�I
�
F__inference_sequential_layer_call_and_return_conditional_losses_243591

inputs
hidden_layer_0_243541
hidden_layer_0_243543
hidden_layer_1_243546
hidden_layer_1_243548
hidden_layer_2_243551
hidden_layer_2_243553
hidden_layer_3_243556
hidden_layer_3_243558
output_layer_243561
output_layer_243563
identity��&Hidden_layer_0/StatefulPartitionedCall�7Hidden_layer_0/kernel/Regularizer/Square/ReadVariableOp�&Hidden_layer_1/StatefulPartitionedCall�7Hidden_layer_1/kernel/Regularizer/Square/ReadVariableOp�&Hidden_layer_2/StatefulPartitionedCall�7Hidden_layer_2/kernel/Regularizer/Square/ReadVariableOp�&Hidden_layer_3/StatefulPartitionedCall�7Hidden_layer_3/kernel/Regularizer/Square/ReadVariableOp�$Output_layer/StatefulPartitionedCall�
&Hidden_layer_0/StatefulPartitionedCallStatefulPartitionedCallinputshidden_layer_0_243541hidden_layer_0_243543*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������P*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_Hidden_layer_0_layer_call_and_return_conditional_losses_2433152(
&Hidden_layer_0/StatefulPartitionedCall�
&Hidden_layer_1/StatefulPartitionedCallStatefulPartitionedCall/Hidden_layer_0/StatefulPartitionedCall:output:0hidden_layer_1_243546hidden_layer_1_243548*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������P*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_Hidden_layer_1_layer_call_and_return_conditional_losses_2433482(
&Hidden_layer_1/StatefulPartitionedCall�
&Hidden_layer_2/StatefulPartitionedCallStatefulPartitionedCall/Hidden_layer_1/StatefulPartitionedCall:output:0hidden_layer_2_243551hidden_layer_2_243553*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������P*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_Hidden_layer_2_layer_call_and_return_conditional_losses_2433812(
&Hidden_layer_2/StatefulPartitionedCall�
&Hidden_layer_3/StatefulPartitionedCallStatefulPartitionedCall/Hidden_layer_2/StatefulPartitionedCall:output:0hidden_layer_3_243556hidden_layer_3_243558*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������P*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_Hidden_layer_3_layer_call_and_return_conditional_losses_2434142(
&Hidden_layer_3/StatefulPartitionedCall�
$Output_layer/StatefulPartitionedCallStatefulPartitionedCall/Hidden_layer_3/StatefulPartitionedCall:output:0output_layer_243561output_layer_243563*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_Output_layer_layer_call_and_return_conditional_losses_2434412&
$Output_layer/StatefulPartitionedCall�
7Hidden_layer_0/kernel/Regularizer/Square/ReadVariableOpReadVariableOphidden_layer_0_243541*
_output_shapes

:P*
dtype029
7Hidden_layer_0/kernel/Regularizer/Square/ReadVariableOp�
(Hidden_layer_0/kernel/Regularizer/SquareSquare?Hidden_layer_0/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:P2*
(Hidden_layer_0/kernel/Regularizer/Square�
'Hidden_layer_0/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2)
'Hidden_layer_0/kernel/Regularizer/Const�
%Hidden_layer_0/kernel/Regularizer/SumSum,Hidden_layer_0/kernel/Regularizer/Square:y:00Hidden_layer_0/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2'
%Hidden_layer_0/kernel/Regularizer/Sum�
'Hidden_layer_0/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2)
'Hidden_layer_0/kernel/Regularizer/mul/x�
%Hidden_layer_0/kernel/Regularizer/mulMul0Hidden_layer_0/kernel/Regularizer/mul/x:output:0.Hidden_layer_0/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2'
%Hidden_layer_0/kernel/Regularizer/mul�
7Hidden_layer_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOphidden_layer_1_243546*
_output_shapes

:PP*
dtype029
7Hidden_layer_1/kernel/Regularizer/Square/ReadVariableOp�
(Hidden_layer_1/kernel/Regularizer/SquareSquare?Hidden_layer_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:PP2*
(Hidden_layer_1/kernel/Regularizer/Square�
'Hidden_layer_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2)
'Hidden_layer_1/kernel/Regularizer/Const�
%Hidden_layer_1/kernel/Regularizer/SumSum,Hidden_layer_1/kernel/Regularizer/Square:y:00Hidden_layer_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2'
%Hidden_layer_1/kernel/Regularizer/Sum�
'Hidden_layer_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2)
'Hidden_layer_1/kernel/Regularizer/mul/x�
%Hidden_layer_1/kernel/Regularizer/mulMul0Hidden_layer_1/kernel/Regularizer/mul/x:output:0.Hidden_layer_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2'
%Hidden_layer_1/kernel/Regularizer/mul�
7Hidden_layer_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOphidden_layer_2_243551*
_output_shapes

:PP*
dtype029
7Hidden_layer_2/kernel/Regularizer/Square/ReadVariableOp�
(Hidden_layer_2/kernel/Regularizer/SquareSquare?Hidden_layer_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:PP2*
(Hidden_layer_2/kernel/Regularizer/Square�
'Hidden_layer_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2)
'Hidden_layer_2/kernel/Regularizer/Const�
%Hidden_layer_2/kernel/Regularizer/SumSum,Hidden_layer_2/kernel/Regularizer/Square:y:00Hidden_layer_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2'
%Hidden_layer_2/kernel/Regularizer/Sum�
'Hidden_layer_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2)
'Hidden_layer_2/kernel/Regularizer/mul/x�
%Hidden_layer_2/kernel/Regularizer/mulMul0Hidden_layer_2/kernel/Regularizer/mul/x:output:0.Hidden_layer_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2'
%Hidden_layer_2/kernel/Regularizer/mul�
7Hidden_layer_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOphidden_layer_3_243556*
_output_shapes

:PP*
dtype029
7Hidden_layer_3/kernel/Regularizer/Square/ReadVariableOp�
(Hidden_layer_3/kernel/Regularizer/SquareSquare?Hidden_layer_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:PP2*
(Hidden_layer_3/kernel/Regularizer/Square�
'Hidden_layer_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2)
'Hidden_layer_3/kernel/Regularizer/Const�
%Hidden_layer_3/kernel/Regularizer/SumSum,Hidden_layer_3/kernel/Regularizer/Square:y:00Hidden_layer_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2'
%Hidden_layer_3/kernel/Regularizer/Sum�
'Hidden_layer_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2)
'Hidden_layer_3/kernel/Regularizer/mul/x�
%Hidden_layer_3/kernel/Regularizer/mulMul0Hidden_layer_3/kernel/Regularizer/mul/x:output:0.Hidden_layer_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2'
%Hidden_layer_3/kernel/Regularizer/mul�
IdentityIdentity-Output_layer/StatefulPartitionedCall:output:0'^Hidden_layer_0/StatefulPartitionedCall8^Hidden_layer_0/kernel/Regularizer/Square/ReadVariableOp'^Hidden_layer_1/StatefulPartitionedCall8^Hidden_layer_1/kernel/Regularizer/Square/ReadVariableOp'^Hidden_layer_2/StatefulPartitionedCall8^Hidden_layer_2/kernel/Regularizer/Square/ReadVariableOp'^Hidden_layer_3/StatefulPartitionedCall8^Hidden_layer_3/kernel/Regularizer/Square/ReadVariableOp%^Output_layer/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:���������::::::::::2P
&Hidden_layer_0/StatefulPartitionedCall&Hidden_layer_0/StatefulPartitionedCall2r
7Hidden_layer_0/kernel/Regularizer/Square/ReadVariableOp7Hidden_layer_0/kernel/Regularizer/Square/ReadVariableOp2P
&Hidden_layer_1/StatefulPartitionedCall&Hidden_layer_1/StatefulPartitionedCall2r
7Hidden_layer_1/kernel/Regularizer/Square/ReadVariableOp7Hidden_layer_1/kernel/Regularizer/Square/ReadVariableOp2P
&Hidden_layer_2/StatefulPartitionedCall&Hidden_layer_2/StatefulPartitionedCall2r
7Hidden_layer_2/kernel/Regularizer/Square/ReadVariableOp7Hidden_layer_2/kernel/Regularizer/Square/ReadVariableOp2P
&Hidden_layer_3/StatefulPartitionedCall&Hidden_layer_3/StatefulPartitionedCall2r
7Hidden_layer_3/kernel/Regularizer/Square/ReadVariableOp7Hidden_layer_3/kernel/Regularizer/Square/ReadVariableOp2L
$Output_layer/StatefulPartitionedCall$Output_layer/StatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
$__inference_signature_wrapper_243751
input_layer
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_layerunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8� **
f%R#
!__inference__wrapped_model_2432942
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:���������::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
'
_output_shapes
:���������
%
_user_specified_nameInput_layer
�
�
+__inference_sequential_layer_call_fn_243902

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_2435912
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:���������::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
J__inference_Hidden_layer_2_layer_call_and_return_conditional_losses_244014

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�7Hidden_layer_2/kernel/Regularizer/Square/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:PP*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:P*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P2	
BiasAddU
EluEluBiasAdd:output:0*
T0*'
_output_shapes
:���������P2
Elu�
7Hidden_layer_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:PP*
dtype029
7Hidden_layer_2/kernel/Regularizer/Square/ReadVariableOp�
(Hidden_layer_2/kernel/Regularizer/SquareSquare?Hidden_layer_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:PP2*
(Hidden_layer_2/kernel/Regularizer/Square�
'Hidden_layer_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2)
'Hidden_layer_2/kernel/Regularizer/Const�
%Hidden_layer_2/kernel/Regularizer/SumSum,Hidden_layer_2/kernel/Regularizer/Square:y:00Hidden_layer_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2'
%Hidden_layer_2/kernel/Regularizer/Sum�
'Hidden_layer_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2)
'Hidden_layer_2/kernel/Regularizer/mul/x�
%Hidden_layer_2/kernel/Regularizer/mulMul0Hidden_layer_2/kernel/Regularizer/mul/x:output:0.Hidden_layer_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2'
%Hidden_layer_2/kernel/Regularizer/mul�
IdentityIdentityElu:activations:0^BiasAdd/ReadVariableOp8^Hidden_layer_2/kernel/Regularizer/Square/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������P2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������P::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2r
7Hidden_layer_2/kernel/Regularizer/Square/ReadVariableOp7Hidden_layer_2/kernel/Regularizer/Square/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������P
 
_user_specified_nameinputs
�
�
/__inference_Hidden_layer_3_layer_call_fn_244055

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������P*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_Hidden_layer_3_layer_call_and_return_conditional_losses_2434142
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������P2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������P::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������P
 
_user_specified_nameinputs
�
�
__inference_loss_fn_3_244119D
@hidden_layer_3_kernel_regularizer_square_readvariableop_resource
identity��7Hidden_layer_3/kernel/Regularizer/Square/ReadVariableOp�
7Hidden_layer_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOp@hidden_layer_3_kernel_regularizer_square_readvariableop_resource*
_output_shapes

:PP*
dtype029
7Hidden_layer_3/kernel/Regularizer/Square/ReadVariableOp�
(Hidden_layer_3/kernel/Regularizer/SquareSquare?Hidden_layer_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:PP2*
(Hidden_layer_3/kernel/Regularizer/Square�
'Hidden_layer_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2)
'Hidden_layer_3/kernel/Regularizer/Const�
%Hidden_layer_3/kernel/Regularizer/SumSum,Hidden_layer_3/kernel/Regularizer/Square:y:00Hidden_layer_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2'
%Hidden_layer_3/kernel/Regularizer/Sum�
'Hidden_layer_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2)
'Hidden_layer_3/kernel/Regularizer/mul/x�
%Hidden_layer_3/kernel/Regularizer/mulMul0Hidden_layer_3/kernel/Regularizer/mul/x:output:0.Hidden_layer_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2'
%Hidden_layer_3/kernel/Regularizer/mul�
IdentityIdentity)Hidden_layer_3/kernel/Regularizer/mul:z:08^Hidden_layer_3/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:2r
7Hidden_layer_3/kernel/Regularizer/Square/ReadVariableOp7Hidden_layer_3/kernel/Regularizer/Square/ReadVariableOp"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
C
Input_layer4
serving_default_Input_layer:0���������@
Output_layer0
StatefulPartitionedCall:0���������tensorflow/serving/predict:��
�5
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
layer_with_weights-4
layer-4
	optimizer
regularization_losses
	variables
	trainable_variables

	keras_api

signatures
*r&call_and_return_all_conditional_losses
s_default_save_signature
t__call__"�2
_tf_keras_sequential�2{"class_name": "Sequential", "name": "sequential", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 24]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "Input_layer"}}, {"class_name": "Dense", "config": {"name": "Hidden_layer_0", "trainable": true, "dtype": "float32", "units": 80, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "Hidden_layer_1", "trainable": true, "dtype": "float32", "units": 80, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "Hidden_layer_2", "trainable": true, "dtype": "float32", "units": 80, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "Hidden_layer_3", "trainable": true, "dtype": "float32", "units": 80, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "Output_layer", "trainable": true, "dtype": "float32", "units": 6, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 24}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 24]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 24]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "Input_layer"}}, {"class_name": "Dense", "config": {"name": "Hidden_layer_0", "trainable": true, "dtype": "float32", "units": 80, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "Hidden_layer_1", "trainable": true, "dtype": "float32", "units": 80, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "Hidden_layer_2", "trainable": true, "dtype": "float32", "units": 80, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "Hidden_layer_3", "trainable": true, "dtype": "float32", "units": 80, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "Output_layer", "trainable": true, "dtype": "float32", "units": 6, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": {"class_name": "MeanAbsolutePercentageError", "config": {"reduction": "auto", "name": "mean_absolute_percentage_error"}}, "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "mean_absolute_error", "dtype": "float32", "fn": "mean_absolute_error"}}, {"class_name": "MeanMetricWrapper", "config": {"name": "mean_absolute_percentage_error", "dtype": "float32", "fn": "mean_absolute_percentage_error"}}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
�

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
*u&call_and_return_all_conditional_losses
v__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "Hidden_layer_0", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Hidden_layer_0", "trainable": true, "dtype": "float32", "units": 80, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 24}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 24]}}
�

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
*w&call_and_return_all_conditional_losses
x__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "Hidden_layer_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Hidden_layer_1", "trainable": true, "dtype": "float32", "units": 80, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 80}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 80]}}
�

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
*y&call_and_return_all_conditional_losses
z__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "Hidden_layer_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Hidden_layer_2", "trainable": true, "dtype": "float32", "units": 80, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 80}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 80]}}
�

kernel
bias
 regularization_losses
!	variables
"trainable_variables
#	keras_api
*{&call_and_return_all_conditional_losses
|__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "Hidden_layer_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Hidden_layer_3", "trainable": true, "dtype": "float32", "units": 80, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 80}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 80]}}
�

$kernel
%bias
&regularization_losses
'	variables
(trainable_variables
)	keras_api
*}&call_and_return_all_conditional_losses
~__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "Output_layer", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Output_layer", "trainable": true, "dtype": "float32", "units": 6, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 80}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 80]}}
�
*iter

+beta_1

,beta_2
	-decay
.learning_ratem^m_m`mambmcmdme$mf%mgvhvivjvkvlvmvnvo$vp%vq"
	optimizer
?
0
�1
�2
�3"
trackable_list_wrapper
f
0
1
2
3
4
5
6
7
$8
%9"
trackable_list_wrapper
f
0
1
2
3
4
5
6
7
$8
%9"
trackable_list_wrapper
�
/metrics
regularization_losses
	variables
0non_trainable_variables

1layers
	trainable_variables
2layer_metrics
3layer_regularization_losses
t__call__
s_default_save_signature
*r&call_and_return_all_conditional_losses
&r"call_and_return_conditional_losses"
_generic_user_object
-
�serving_default"
signature_map
':%P2Hidden_layer_0/kernel
!:P2Hidden_layer_0/bias
'
0"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
�
4metrics
regularization_losses
	variables
5non_trainable_variables

6layers
trainable_variables
7layer_metrics
8layer_regularization_losses
v__call__
*u&call_and_return_all_conditional_losses
&u"call_and_return_conditional_losses"
_generic_user_object
':%PP2Hidden_layer_1/kernel
!:P2Hidden_layer_1/bias
(
�0"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
�
9metrics
regularization_losses
	variables
:non_trainable_variables

;layers
trainable_variables
<layer_metrics
=layer_regularization_losses
x__call__
*w&call_and_return_all_conditional_losses
&w"call_and_return_conditional_losses"
_generic_user_object
':%PP2Hidden_layer_2/kernel
!:P2Hidden_layer_2/bias
(
�0"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
�
>metrics
regularization_losses
	variables
?non_trainable_variables

@layers
trainable_variables
Alayer_metrics
Blayer_regularization_losses
z__call__
*y&call_and_return_all_conditional_losses
&y"call_and_return_conditional_losses"
_generic_user_object
':%PP2Hidden_layer_3/kernel
!:P2Hidden_layer_3/bias
(
�0"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
�
Cmetrics
 regularization_losses
!	variables
Dnon_trainable_variables

Elayers
"trainable_variables
Flayer_metrics
Glayer_regularization_losses
|__call__
*{&call_and_return_all_conditional_losses
&{"call_and_return_conditional_losses"
_generic_user_object
%:#P2Output_layer/kernel
:2Output_layer/bias
 "
trackable_list_wrapper
.
$0
%1"
trackable_list_wrapper
.
$0
%1"
trackable_list_wrapper
�
Hmetrics
&regularization_losses
'	variables
Inon_trainable_variables

Jlayers
(trainable_variables
Klayer_metrics
Llayer_regularization_losses
~__call__
*}&call_and_return_all_conditional_losses
&}"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
5
M0
N1
O2"
trackable_list_wrapper
 "
trackable_list_wrapper
C
0
1
2
3
4"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
�
	Ptotal
	Qcount
R	variables
S	keras_api"�
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
�
	Ttotal
	Ucount
V
_fn_kwargs
W	variables
X	keras_api"�
_tf_keras_metric�{"class_name": "MeanMetricWrapper", "name": "mean_absolute_error", "dtype": "float32", "config": {"name": "mean_absolute_error", "dtype": "float32", "fn": "mean_absolute_error"}}
�
	Ytotal
	Zcount
[
_fn_kwargs
\	variables
]	keras_api"�
_tf_keras_metric�{"class_name": "MeanMetricWrapper", "name": "mean_absolute_percentage_error", "dtype": "float32", "config": {"name": "mean_absolute_percentage_error", "dtype": "float32", "fn": "mean_absolute_percentage_error"}}
:  (2total
:  (2count
.
P0
Q1"
trackable_list_wrapper
-
R	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
T0
U1"
trackable_list_wrapper
-
W	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
Y0
Z1"
trackable_list_wrapper
-
\	variables"
_generic_user_object
,:*P2Adam/Hidden_layer_0/kernel/m
&:$P2Adam/Hidden_layer_0/bias/m
,:*PP2Adam/Hidden_layer_1/kernel/m
&:$P2Adam/Hidden_layer_1/bias/m
,:*PP2Adam/Hidden_layer_2/kernel/m
&:$P2Adam/Hidden_layer_2/bias/m
,:*PP2Adam/Hidden_layer_3/kernel/m
&:$P2Adam/Hidden_layer_3/bias/m
*:(P2Adam/Output_layer/kernel/m
$:"2Adam/Output_layer/bias/m
,:*P2Adam/Hidden_layer_0/kernel/v
&:$P2Adam/Hidden_layer_0/bias/v
,:*PP2Adam/Hidden_layer_1/kernel/v
&:$P2Adam/Hidden_layer_1/bias/v
,:*PP2Adam/Hidden_layer_2/kernel/v
&:$P2Adam/Hidden_layer_2/bias/v
,:*PP2Adam/Hidden_layer_3/kernel/v
&:$P2Adam/Hidden_layer_3/bias/v
*:(P2Adam/Output_layer/kernel/v
$:"2Adam/Output_layer/bias/v
�2�
F__inference_sequential_layer_call_and_return_conditional_losses_243814
F__inference_sequential_layer_call_and_return_conditional_losses_243877
F__inference_sequential_layer_call_and_return_conditional_losses_243535
F__inference_sequential_layer_call_and_return_conditional_losses_243482�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
!__inference__wrapped_model_243294�
���
FullArgSpec
args� 
varargsjargs
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� **�'
%�"
Input_layer���������
�2�
+__inference_sequential_layer_call_fn_243902
+__inference_sequential_layer_call_fn_243692
+__inference_sequential_layer_call_fn_243614
+__inference_sequential_layer_call_fn_243927�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
J__inference_Hidden_layer_0_layer_call_and_return_conditional_losses_243950�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
/__inference_Hidden_layer_0_layer_call_fn_243959�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
J__inference_Hidden_layer_1_layer_call_and_return_conditional_losses_243982�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
/__inference_Hidden_layer_1_layer_call_fn_243991�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
J__inference_Hidden_layer_2_layer_call_and_return_conditional_losses_244014�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
/__inference_Hidden_layer_2_layer_call_fn_244023�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
J__inference_Hidden_layer_3_layer_call_and_return_conditional_losses_244046�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
/__inference_Hidden_layer_3_layer_call_fn_244055�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
H__inference_Output_layer_layer_call_and_return_conditional_losses_244066�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
-__inference_Output_layer_layer_call_fn_244075�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
__inference_loss_fn_0_244086�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�2�
__inference_loss_fn_1_244097�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�2�
__inference_loss_fn_2_244108�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�2�
__inference_loss_fn_3_244119�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�B�
$__inference_signature_wrapper_243751Input_layer"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 �
J__inference_Hidden_layer_0_layer_call_and_return_conditional_losses_243950\/�,
%�"
 �
inputs���������
� "%�"
�
0���������P
� �
/__inference_Hidden_layer_0_layer_call_fn_243959O/�,
%�"
 �
inputs���������
� "����������P�
J__inference_Hidden_layer_1_layer_call_and_return_conditional_losses_243982\/�,
%�"
 �
inputs���������P
� "%�"
�
0���������P
� �
/__inference_Hidden_layer_1_layer_call_fn_243991O/�,
%�"
 �
inputs���������P
� "����������P�
J__inference_Hidden_layer_2_layer_call_and_return_conditional_losses_244014\/�,
%�"
 �
inputs���������P
� "%�"
�
0���������P
� �
/__inference_Hidden_layer_2_layer_call_fn_244023O/�,
%�"
 �
inputs���������P
� "����������P�
J__inference_Hidden_layer_3_layer_call_and_return_conditional_losses_244046\/�,
%�"
 �
inputs���������P
� "%�"
�
0���������P
� �
/__inference_Hidden_layer_3_layer_call_fn_244055O/�,
%�"
 �
inputs���������P
� "����������P�
H__inference_Output_layer_layer_call_and_return_conditional_losses_244066\$%/�,
%�"
 �
inputs���������P
� "%�"
�
0���������
� �
-__inference_Output_layer_layer_call_fn_244075O$%/�,
%�"
 �
inputs���������P
� "�����������
!__inference__wrapped_model_243294
$%4�1
*�'
%�"
Input_layer���������
� ";�8
6
Output_layer&�#
Output_layer���������;
__inference_loss_fn_0_244086�

� 
� "� ;
__inference_loss_fn_1_244097�

� 
� "� ;
__inference_loss_fn_2_244108�

� 
� "� ;
__inference_loss_fn_3_244119�

� 
� "� �
F__inference_sequential_layer_call_and_return_conditional_losses_243482q
$%<�9
2�/
%�"
Input_layer���������
p

 
� "%�"
�
0���������
� �
F__inference_sequential_layer_call_and_return_conditional_losses_243535q
$%<�9
2�/
%�"
Input_layer���������
p 

 
� "%�"
�
0���������
� �
F__inference_sequential_layer_call_and_return_conditional_losses_243814l
$%7�4
-�*
 �
inputs���������
p

 
� "%�"
�
0���������
� �
F__inference_sequential_layer_call_and_return_conditional_losses_243877l
$%7�4
-�*
 �
inputs���������
p 

 
� "%�"
�
0���������
� �
+__inference_sequential_layer_call_fn_243614d
$%<�9
2�/
%�"
Input_layer���������
p

 
� "�����������
+__inference_sequential_layer_call_fn_243692d
$%<�9
2�/
%�"
Input_layer���������
p 

 
� "�����������
+__inference_sequential_layer_call_fn_243902_
$%7�4
-�*
 �
inputs���������
p

 
� "�����������
+__inference_sequential_layer_call_fn_243927_
$%7�4
-�*
 �
inputs���������
p 

 
� "�����������
$__inference_signature_wrapper_243751�
$%C�@
� 
9�6
4
Input_layer%�"
Input_layer���������";�8
6
Output_layer&�#
Output_layer���������