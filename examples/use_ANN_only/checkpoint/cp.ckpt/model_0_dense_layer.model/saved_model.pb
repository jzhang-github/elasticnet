��
��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
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
�
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( �
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
executor_typestring ��
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
 �"serve*2.10.02v2.10.0-rc3-6-g359c3cdfc5f8��
�
Adam/output_layer/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/output_layer/bias/v
�
,Adam/output_layer/bias/v/Read/ReadVariableOpReadVariableOpAdam/output_layer/bias/v*
_output_shapes
:*
dtype0
�
Adam/output_layer/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*+
shared_nameAdam/output_layer/kernel/v
�
.Adam/output_layer/kernel/v/Read/ReadVariableOpReadVariableOpAdam/output_layer/kernel/v*
_output_shapes

:*
dtype0
�
Adam/Hidden_layer_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nameAdam/Hidden_layer_2/bias/v
�
.Adam/Hidden_layer_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/Hidden_layer_2/bias/v*
_output_shapes
:*
dtype0
�
Adam/Hidden_layer_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*-
shared_nameAdam/Hidden_layer_2/kernel/v
�
0Adam/Hidden_layer_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/Hidden_layer_2/kernel/v*
_output_shapes

:*
dtype0
�
Adam/Hidden_layer_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nameAdam/Hidden_layer_1/bias/v
�
.Adam/Hidden_layer_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/Hidden_layer_1/bias/v*
_output_shapes
:*
dtype0
�
Adam/Hidden_layer_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*-
shared_nameAdam/Hidden_layer_1/kernel/v
�
0Adam/Hidden_layer_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/Hidden_layer_1/kernel/v*
_output_shapes

:*
dtype0
�
Adam/Hidden_layer_0/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nameAdam/Hidden_layer_0/bias/v
�
.Adam/Hidden_layer_0/bias/v/Read/ReadVariableOpReadVariableOpAdam/Hidden_layer_0/bias/v*
_output_shapes
:*
dtype0
�
Adam/Hidden_layer_0/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*-
shared_nameAdam/Hidden_layer_0/kernel/v
�
0Adam/Hidden_layer_0/kernel/v/Read/ReadVariableOpReadVariableOpAdam/Hidden_layer_0/kernel/v*
_output_shapes

:*
dtype0
�
Adam/output_layer/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/output_layer/bias/m
�
,Adam/output_layer/bias/m/Read/ReadVariableOpReadVariableOpAdam/output_layer/bias/m*
_output_shapes
:*
dtype0
�
Adam/output_layer/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*+
shared_nameAdam/output_layer/kernel/m
�
.Adam/output_layer/kernel/m/Read/ReadVariableOpReadVariableOpAdam/output_layer/kernel/m*
_output_shapes

:*
dtype0
�
Adam/Hidden_layer_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nameAdam/Hidden_layer_2/bias/m
�
.Adam/Hidden_layer_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/Hidden_layer_2/bias/m*
_output_shapes
:*
dtype0
�
Adam/Hidden_layer_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*-
shared_nameAdam/Hidden_layer_2/kernel/m
�
0Adam/Hidden_layer_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/Hidden_layer_2/kernel/m*
_output_shapes

:*
dtype0
�
Adam/Hidden_layer_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nameAdam/Hidden_layer_1/bias/m
�
.Adam/Hidden_layer_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/Hidden_layer_1/bias/m*
_output_shapes
:*
dtype0
�
Adam/Hidden_layer_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*-
shared_nameAdam/Hidden_layer_1/kernel/m
�
0Adam/Hidden_layer_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/Hidden_layer_1/kernel/m*
_output_shapes

:*
dtype0
�
Adam/Hidden_layer_0/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nameAdam/Hidden_layer_0/bias/m
�
.Adam/Hidden_layer_0/bias/m/Read/ReadVariableOpReadVariableOpAdam/Hidden_layer_0/bias/m*
_output_shapes
:*
dtype0
�
Adam/Hidden_layer_0/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*-
shared_nameAdam/Hidden_layer_0/kernel/m
�
0Adam/Hidden_layer_0/kernel/m/Read/ReadVariableOpReadVariableOpAdam/Hidden_layer_0/kernel/m*
_output_shapes

:*
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
z
output_layer/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameoutput_layer/bias
s
%output_layer/bias/Read/ReadVariableOpReadVariableOpoutput_layer/bias*
_output_shapes
:*
dtype0
�
output_layer/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*$
shared_nameoutput_layer/kernel
{
'output_layer/kernel/Read/ReadVariableOpReadVariableOpoutput_layer/kernel*
_output_shapes

:*
dtype0
~
Hidden_layer_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameHidden_layer_2/bias
w
'Hidden_layer_2/bias/Read/ReadVariableOpReadVariableOpHidden_layer_2/bias*
_output_shapes
:*
dtype0
�
Hidden_layer_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*&
shared_nameHidden_layer_2/kernel

)Hidden_layer_2/kernel/Read/ReadVariableOpReadVariableOpHidden_layer_2/kernel*
_output_shapes

:*
dtype0
~
Hidden_layer_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameHidden_layer_1/bias
w
'Hidden_layer_1/bias/Read/ReadVariableOpReadVariableOpHidden_layer_1/bias*
_output_shapes
:*
dtype0
�
Hidden_layer_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*&
shared_nameHidden_layer_1/kernel

)Hidden_layer_1/kernel/Read/ReadVariableOpReadVariableOpHidden_layer_1/kernel*
_output_shapes

:*
dtype0
~
Hidden_layer_0/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameHidden_layer_0/bias
w
'Hidden_layer_0/bias/Read/ReadVariableOpReadVariableOpHidden_layer_0/bias*
_output_shapes
:*
dtype0
�
Hidden_layer_0/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*&
shared_nameHidden_layer_0/kernel

)Hidden_layer_0/kernel/Read/ReadVariableOpReadVariableOpHidden_layer_0/kernel*
_output_shapes

:*
dtype0
~
serving_default_input_layerPlaceholder*'
_output_shapes
:���������*
dtype0*
shape:���������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_layerHidden_layer_0/kernelHidden_layer_0/biasHidden_layer_1/kernelHidden_layer_1/biasHidden_layer_2/kernelHidden_layer_2/biasoutput_layer/kerneloutput_layer/bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *,
f'R%
#__inference_signature_wrapper_34503

NoOpNoOp
�>
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�=
value�=B�= B�=
�
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
	variables
trainable_variables
regularization_losses
	keras_api
	__call__
*
&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures*
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias*
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias*
�
	variables
trainable_variables
 regularization_losses
!	keras_api
"__call__
*#&call_and_return_all_conditional_losses

$kernel
%bias*
�
&	variables
'trainable_variables
(regularization_losses
)	keras_api
*__call__
*+&call_and_return_all_conditional_losses

,kernel
-bias*
<
0
1
2
3
$4
%5
,6
-7*
<
0
1
2
3
$4
%5
,6
-7*

.0
/1
02* 
�
1non_trainable_variables

2layers
3metrics
4layer_regularization_losses
5layer_metrics
	variables
trainable_variables
regularization_losses
	__call__
_default_save_signature
*
&call_and_return_all_conditional_losses
&
"call_and_return_conditional_losses*
6
6trace_0
7trace_1
8trace_2
9trace_3* 
6
:trace_0
;trace_1
<trace_2
=trace_3* 
* 
�
>iter

?beta_1

@beta_2
	Adecay
Blearning_ratemtmumvmw$mx%my,mz-m{v|v}v~v$v�%v�,v�-v�*

Cserving_default* 

0
1*

0
1*
	
.0* 
�
Dnon_trainable_variables

Elayers
Fmetrics
Glayer_regularization_losses
Hlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

Itrace_0* 

Jtrace_0* 
e_
VARIABLE_VALUEHidden_layer_0/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEHidden_layer_0/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*
	
/0* 
�
Knon_trainable_variables

Llayers
Mmetrics
Nlayer_regularization_losses
Olayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

Ptrace_0* 

Qtrace_0* 
e_
VARIABLE_VALUEHidden_layer_1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEHidden_layer_1/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*

$0
%1*

$0
%1*
	
00* 
�
Rnon_trainable_variables

Slayers
Tmetrics
Ulayer_regularization_losses
Vlayer_metrics
	variables
trainable_variables
 regularization_losses
"__call__
*#&call_and_return_all_conditional_losses
&#"call_and_return_conditional_losses*

Wtrace_0* 

Xtrace_0* 
e_
VARIABLE_VALUEHidden_layer_2/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEHidden_layer_2/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*

,0
-1*

,0
-1*
* 
�
Ynon_trainable_variables

Zlayers
[metrics
\layer_regularization_losses
]layer_metrics
&	variables
'trainable_variables
(regularization_losses
*__call__
*+&call_and_return_all_conditional_losses
&+"call_and_return_conditional_losses*

^trace_0* 

_trace_0* 
c]
VARIABLE_VALUEoutput_layer/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEoutput_layer/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*

`trace_0* 

atrace_0* 

btrace_0* 
* 
 
0
1
2
3*

c0
d1
e2*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
	
.0* 
* 
* 
* 
* 
* 
* 
	
/0* 
* 
* 
* 
* 
* 
* 
	
00* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
8
f	variables
g	keras_api
	htotal
	icount*
H
j	variables
k	keras_api
	ltotal
	mcount
n
_fn_kwargs*
H
o	variables
p	keras_api
	qtotal
	rcount
s
_fn_kwargs*

h0
i1*

f	variables*
UO
VARIABLE_VALUEtotal_24keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_24keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

l0
m1*

j	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 

q0
r1*

o	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
��
VARIABLE_VALUEAdam/Hidden_layer_0/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�~
VARIABLE_VALUEAdam/Hidden_layer_0/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEAdam/Hidden_layer_1/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�~
VARIABLE_VALUEAdam/Hidden_layer_1/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEAdam/Hidden_layer_2/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�~
VARIABLE_VALUEAdam/Hidden_layer_2/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEAdam/output_layer/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�|
VARIABLE_VALUEAdam/output_layer/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEAdam/Hidden_layer_0/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
�~
VARIABLE_VALUEAdam/Hidden_layer_0/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEAdam/Hidden_layer_1/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
�~
VARIABLE_VALUEAdam/Hidden_layer_1/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEAdam/Hidden_layer_2/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
�~
VARIABLE_VALUEAdam/Hidden_layer_2/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEAdam/output_layer/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
�|
VARIABLE_VALUEAdam/output_layer/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename)Hidden_layer_0/kernel/Read/ReadVariableOp'Hidden_layer_0/bias/Read/ReadVariableOp)Hidden_layer_1/kernel/Read/ReadVariableOp'Hidden_layer_1/bias/Read/ReadVariableOp)Hidden_layer_2/kernel/Read/ReadVariableOp'Hidden_layer_2/bias/Read/ReadVariableOp'output_layer/kernel/Read/ReadVariableOp%output_layer/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal_2/Read/ReadVariableOpcount_2/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp0Adam/Hidden_layer_0/kernel/m/Read/ReadVariableOp.Adam/Hidden_layer_0/bias/m/Read/ReadVariableOp0Adam/Hidden_layer_1/kernel/m/Read/ReadVariableOp.Adam/Hidden_layer_1/bias/m/Read/ReadVariableOp0Adam/Hidden_layer_2/kernel/m/Read/ReadVariableOp.Adam/Hidden_layer_2/bias/m/Read/ReadVariableOp.Adam/output_layer/kernel/m/Read/ReadVariableOp,Adam/output_layer/bias/m/Read/ReadVariableOp0Adam/Hidden_layer_0/kernel/v/Read/ReadVariableOp.Adam/Hidden_layer_0/bias/v/Read/ReadVariableOp0Adam/Hidden_layer_1/kernel/v/Read/ReadVariableOp.Adam/Hidden_layer_1/bias/v/Read/ReadVariableOp0Adam/Hidden_layer_2/kernel/v/Read/ReadVariableOp.Adam/Hidden_layer_2/bias/v/Read/ReadVariableOp.Adam/output_layer/kernel/v/Read/ReadVariableOp,Adam/output_layer/bias/v/Read/ReadVariableOpConst*0
Tin)
'2%	*
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
GPU 2J 8� *'
f"R 
__inference__traced_save_34892
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameHidden_layer_0/kernelHidden_layer_0/biasHidden_layer_1/kernelHidden_layer_1/biasHidden_layer_2/kernelHidden_layer_2/biasoutput_layer/kerneloutput_layer/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotal_2count_2total_1count_1totalcountAdam/Hidden_layer_0/kernel/mAdam/Hidden_layer_0/bias/mAdam/Hidden_layer_1/kernel/mAdam/Hidden_layer_1/bias/mAdam/Hidden_layer_2/kernel/mAdam/Hidden_layer_2/bias/mAdam/output_layer/kernel/mAdam/output_layer/bias/mAdam/Hidden_layer_0/kernel/vAdam/Hidden_layer_0/bias/vAdam/Hidden_layer_1/kernel/vAdam/Hidden_layer_1/bias/vAdam/Hidden_layer_2/kernel/vAdam/Hidden_layer_2/bias/vAdam/output_layer/kernel/vAdam/output_layer/bias/v*/
Tin(
&2$*
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
GPU 2J 8� **
f%R#
!__inference__traced_restore_35007��
�
�
I__inference_Hidden_layer_1_layer_call_and_return_conditional_losses_34175

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�7Hidden_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������N
EluEluBiasAdd:output:0*
T0*'
_output_shapes
:����������
7Hidden_layer_1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0�
(Hidden_layer_1/kernel/Regularizer/L2LossL2Loss?Hidden_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: l
'Hidden_layer_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
%Hidden_layer_1/kernel/Regularizer/mulMul0Hidden_layer_1/kernel/Regularizer/mul/x:output:01Hidden_layer_1/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: `
IdentityIdentityElu:activations:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^BiasAdd/ReadVariableOp8^Hidden_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2r
7Hidden_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp7Hidden_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�

�
G__inference_output_layer_layer_call_and_return_conditional_losses_34737

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������N
EluEluBiasAdd:output:0*
T0*'
_output_shapes
:���������`
IdentityIdentityElu:activations:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
I__inference_Hidden_layer_0_layer_call_and_return_conditional_losses_34154

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�7Hidden_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������N
EluEluBiasAdd:output:0*
T0*'
_output_shapes
:����������
7Hidden_layer_0/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0�
(Hidden_layer_0/kernel/Regularizer/L2LossL2Loss?Hidden_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: l
'Hidden_layer_0/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
%Hidden_layer_0/kernel/Regularizer/mulMul0Hidden_layer_0/kernel/Regularizer/mul/x:output:01Hidden_layer_0/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: `
IdentityIdentityElu:activations:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^BiasAdd/ReadVariableOp8^Hidden_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2r
7Hidden_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp7Hidden_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�	
�
__inference_loss_fn_1_34755R
@hidden_layer_1_kernel_regularizer_l2loss_readvariableop_resource:
identity��7Hidden_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp�
7Hidden_layer_1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp@hidden_layer_1_kernel_regularizer_l2loss_readvariableop_resource*
_output_shapes

:*
dtype0�
(Hidden_layer_1/kernel/Regularizer/L2LossL2Loss?Hidden_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: l
'Hidden_layer_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
%Hidden_layer_1/kernel/Regularizer/mulMul0Hidden_layer_1/kernel/Regularizer/mul/x:output:01Hidden_layer_1/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: g
IdentityIdentity)Hidden_layer_1/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOp8^Hidden_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2r
7Hidden_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp7Hidden_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp
�	
�
__inference_loss_fn_0_34746R
@hidden_layer_0_kernel_regularizer_l2loss_readvariableop_resource:
identity��7Hidden_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp�
7Hidden_layer_0/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp@hidden_layer_0_kernel_regularizer_l2loss_readvariableop_resource*
_output_shapes

:*
dtype0�
(Hidden_layer_0/kernel/Regularizer/L2LossL2Loss?Hidden_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: l
'Hidden_layer_0/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
%Hidden_layer_0/kernel/Regularizer/mulMul0Hidden_layer_0/kernel/Regularizer/mul/x:output:01Hidden_layer_0/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: g
IdentityIdentity)Hidden_layer_0/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOp8^Hidden_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2r
7Hidden_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp7Hidden_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp
�	
�
*__inference_sequential_layer_call_fn_34390
input_layer
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_layerunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_34350o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
'
_output_shapes
:���������
%
_user_specified_nameinput_layer
�	
�
#__inference_signature_wrapper_34503
input_layer
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_layerunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *)
f$R"
 __inference__wrapped_model_34132o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
'
_output_shapes
:���������
%
_user_specified_nameinput_layer
�
�
,__inference_output_layer_layer_call_fn_34726

inputs
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_output_layer_layer_call_and_return_conditional_losses_34213o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
I__inference_Hidden_layer_1_layer_call_and_return_conditional_losses_34693

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�7Hidden_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������N
EluEluBiasAdd:output:0*
T0*'
_output_shapes
:����������
7Hidden_layer_1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0�
(Hidden_layer_1/kernel/Regularizer/L2LossL2Loss?Hidden_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: l
'Hidden_layer_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
%Hidden_layer_1/kernel/Regularizer/mulMul0Hidden_layer_1/kernel/Regularizer/mul/x:output:01Hidden_layer_1/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: `
IdentityIdentityElu:activations:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^BiasAdd/ReadVariableOp8^Hidden_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2r
7Hidden_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp7Hidden_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�	
�
__inference_loss_fn_2_34764R
@hidden_layer_2_kernel_regularizer_l2loss_readvariableop_resource:
identity��7Hidden_layer_2/kernel/Regularizer/L2Loss/ReadVariableOp�
7Hidden_layer_2/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp@hidden_layer_2_kernel_regularizer_l2loss_readvariableop_resource*
_output_shapes

:*
dtype0�
(Hidden_layer_2/kernel/Regularizer/L2LossL2Loss?Hidden_layer_2/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: l
'Hidden_layer_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
%Hidden_layer_2/kernel/Regularizer/mulMul0Hidden_layer_2/kernel/Regularizer/mul/x:output:01Hidden_layer_2/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: g
IdentityIdentity)Hidden_layer_2/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOp8^Hidden_layer_2/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2r
7Hidden_layer_2/kernel/Regularizer/L2Loss/ReadVariableOp7Hidden_layer_2/kernel/Regularizer/L2Loss/ReadVariableOp
�
�
.__inference_Hidden_layer_1_layer_call_fn_34678

inputs
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_Hidden_layer_1_layer_call_and_return_conditional_losses_34175o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�+
�
E__inference_sequential_layer_call_and_return_conditional_losses_34426
input_layer&
hidden_layer_0_34393:"
hidden_layer_0_34395:&
hidden_layer_1_34398:"
hidden_layer_1_34400:&
hidden_layer_2_34403:"
hidden_layer_2_34405:$
output_layer_34408: 
output_layer_34410:
identity��&Hidden_layer_0/StatefulPartitionedCall�7Hidden_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp�&Hidden_layer_1/StatefulPartitionedCall�7Hidden_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp�&Hidden_layer_2/StatefulPartitionedCall�7Hidden_layer_2/kernel/Regularizer/L2Loss/ReadVariableOp�$output_layer/StatefulPartitionedCall�
&Hidden_layer_0/StatefulPartitionedCallStatefulPartitionedCallinput_layerhidden_layer_0_34393hidden_layer_0_34395*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_Hidden_layer_0_layer_call_and_return_conditional_losses_34154�
&Hidden_layer_1/StatefulPartitionedCallStatefulPartitionedCall/Hidden_layer_0/StatefulPartitionedCall:output:0hidden_layer_1_34398hidden_layer_1_34400*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_Hidden_layer_1_layer_call_and_return_conditional_losses_34175�
&Hidden_layer_2/StatefulPartitionedCallStatefulPartitionedCall/Hidden_layer_1/StatefulPartitionedCall:output:0hidden_layer_2_34403hidden_layer_2_34405*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_Hidden_layer_2_layer_call_and_return_conditional_losses_34196�
$output_layer/StatefulPartitionedCallStatefulPartitionedCall/Hidden_layer_2/StatefulPartitionedCall:output:0output_layer_34408output_layer_34410*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_output_layer_layer_call_and_return_conditional_losses_34213�
7Hidden_layer_0/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOphidden_layer_0_34393*
_output_shapes

:*
dtype0�
(Hidden_layer_0/kernel/Regularizer/L2LossL2Loss?Hidden_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: l
'Hidden_layer_0/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
%Hidden_layer_0/kernel/Regularizer/mulMul0Hidden_layer_0/kernel/Regularizer/mul/x:output:01Hidden_layer_0/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
7Hidden_layer_1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOphidden_layer_1_34398*
_output_shapes

:*
dtype0�
(Hidden_layer_1/kernel/Regularizer/L2LossL2Loss?Hidden_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: l
'Hidden_layer_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
%Hidden_layer_1/kernel/Regularizer/mulMul0Hidden_layer_1/kernel/Regularizer/mul/x:output:01Hidden_layer_1/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
7Hidden_layer_2/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOphidden_layer_2_34403*
_output_shapes

:*
dtype0�
(Hidden_layer_2/kernel/Regularizer/L2LossL2Loss?Hidden_layer_2/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: l
'Hidden_layer_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
%Hidden_layer_2/kernel/Regularizer/mulMul0Hidden_layer_2/kernel/Regularizer/mul/x:output:01Hidden_layer_2/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: |
IdentityIdentity-output_layer/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp'^Hidden_layer_0/StatefulPartitionedCall8^Hidden_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp'^Hidden_layer_1/StatefulPartitionedCall8^Hidden_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp'^Hidden_layer_2/StatefulPartitionedCall8^Hidden_layer_2/kernel/Regularizer/L2Loss/ReadVariableOp%^output_layer/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : : : : : : : 2P
&Hidden_layer_0/StatefulPartitionedCall&Hidden_layer_0/StatefulPartitionedCall2r
7Hidden_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp7Hidden_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp2P
&Hidden_layer_1/StatefulPartitionedCall&Hidden_layer_1/StatefulPartitionedCall2r
7Hidden_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp7Hidden_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp2P
&Hidden_layer_2/StatefulPartitionedCall&Hidden_layer_2/StatefulPartitionedCall2r
7Hidden_layer_2/kernel/Regularizer/L2Loss/ReadVariableOp7Hidden_layer_2/kernel/Regularizer/L2Loss/ReadVariableOp2L
$output_layer/StatefulPartitionedCall$output_layer/StatefulPartitionedCall:T P
'
_output_shapes
:���������
%
_user_specified_nameinput_layer
�;
�
E__inference_sequential_layer_call_and_return_conditional_losses_34601

inputs?
-hidden_layer_0_matmul_readvariableop_resource:<
.hidden_layer_0_biasadd_readvariableop_resource:?
-hidden_layer_1_matmul_readvariableop_resource:<
.hidden_layer_1_biasadd_readvariableop_resource:?
-hidden_layer_2_matmul_readvariableop_resource:<
.hidden_layer_2_biasadd_readvariableop_resource:=
+output_layer_matmul_readvariableop_resource::
,output_layer_biasadd_readvariableop_resource:
identity��%Hidden_layer_0/BiasAdd/ReadVariableOp�$Hidden_layer_0/MatMul/ReadVariableOp�7Hidden_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp�%Hidden_layer_1/BiasAdd/ReadVariableOp�$Hidden_layer_1/MatMul/ReadVariableOp�7Hidden_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp�%Hidden_layer_2/BiasAdd/ReadVariableOp�$Hidden_layer_2/MatMul/ReadVariableOp�7Hidden_layer_2/kernel/Regularizer/L2Loss/ReadVariableOp�#output_layer/BiasAdd/ReadVariableOp�"output_layer/MatMul/ReadVariableOp�
$Hidden_layer_0/MatMul/ReadVariableOpReadVariableOp-hidden_layer_0_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
Hidden_layer_0/MatMulMatMulinputs,Hidden_layer_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
%Hidden_layer_0/BiasAdd/ReadVariableOpReadVariableOp.hidden_layer_0_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
Hidden_layer_0/BiasAddBiasAddHidden_layer_0/MatMul:product:0-Hidden_layer_0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������l
Hidden_layer_0/EluEluHidden_layer_0/BiasAdd:output:0*
T0*'
_output_shapes
:����������
$Hidden_layer_1/MatMul/ReadVariableOpReadVariableOp-hidden_layer_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
Hidden_layer_1/MatMulMatMul Hidden_layer_0/Elu:activations:0,Hidden_layer_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
%Hidden_layer_1/BiasAdd/ReadVariableOpReadVariableOp.hidden_layer_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
Hidden_layer_1/BiasAddBiasAddHidden_layer_1/MatMul:product:0-Hidden_layer_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������l
Hidden_layer_1/EluEluHidden_layer_1/BiasAdd:output:0*
T0*'
_output_shapes
:����������
$Hidden_layer_2/MatMul/ReadVariableOpReadVariableOp-hidden_layer_2_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
Hidden_layer_2/MatMulMatMul Hidden_layer_1/Elu:activations:0,Hidden_layer_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
%Hidden_layer_2/BiasAdd/ReadVariableOpReadVariableOp.hidden_layer_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
Hidden_layer_2/BiasAddBiasAddHidden_layer_2/MatMul:product:0-Hidden_layer_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������l
Hidden_layer_2/EluEluHidden_layer_2/BiasAdd:output:0*
T0*'
_output_shapes
:����������
"output_layer/MatMul/ReadVariableOpReadVariableOp+output_layer_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
output_layer/MatMulMatMul Hidden_layer_2/Elu:activations:0*output_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
#output_layer/BiasAdd/ReadVariableOpReadVariableOp,output_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
output_layer/BiasAddBiasAddoutput_layer/MatMul:product:0+output_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������h
output_layer/EluEluoutput_layer/BiasAdd:output:0*
T0*'
_output_shapes
:����������
7Hidden_layer_0/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp-hidden_layer_0_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
(Hidden_layer_0/kernel/Regularizer/L2LossL2Loss?Hidden_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: l
'Hidden_layer_0/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
%Hidden_layer_0/kernel/Regularizer/mulMul0Hidden_layer_0/kernel/Regularizer/mul/x:output:01Hidden_layer_0/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
7Hidden_layer_1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp-hidden_layer_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
(Hidden_layer_1/kernel/Regularizer/L2LossL2Loss?Hidden_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: l
'Hidden_layer_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
%Hidden_layer_1/kernel/Regularizer/mulMul0Hidden_layer_1/kernel/Regularizer/mul/x:output:01Hidden_layer_1/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
7Hidden_layer_2/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp-hidden_layer_2_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
(Hidden_layer_2/kernel/Regularizer/L2LossL2Loss?Hidden_layer_2/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: l
'Hidden_layer_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
%Hidden_layer_2/kernel/Regularizer/mulMul0Hidden_layer_2/kernel/Regularizer/mul/x:output:01Hidden_layer_2/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: m
IdentityIdentityoutput_layer/Elu:activations:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp&^Hidden_layer_0/BiasAdd/ReadVariableOp%^Hidden_layer_0/MatMul/ReadVariableOp8^Hidden_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp&^Hidden_layer_1/BiasAdd/ReadVariableOp%^Hidden_layer_1/MatMul/ReadVariableOp8^Hidden_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp&^Hidden_layer_2/BiasAdd/ReadVariableOp%^Hidden_layer_2/MatMul/ReadVariableOp8^Hidden_layer_2/kernel/Regularizer/L2Loss/ReadVariableOp$^output_layer/BiasAdd/ReadVariableOp#^output_layer/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : : : : : : : 2N
%Hidden_layer_0/BiasAdd/ReadVariableOp%Hidden_layer_0/BiasAdd/ReadVariableOp2L
$Hidden_layer_0/MatMul/ReadVariableOp$Hidden_layer_0/MatMul/ReadVariableOp2r
7Hidden_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp7Hidden_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp2N
%Hidden_layer_1/BiasAdd/ReadVariableOp%Hidden_layer_1/BiasAdd/ReadVariableOp2L
$Hidden_layer_1/MatMul/ReadVariableOp$Hidden_layer_1/MatMul/ReadVariableOp2r
7Hidden_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp7Hidden_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp2N
%Hidden_layer_2/BiasAdd/ReadVariableOp%Hidden_layer_2/BiasAdd/ReadVariableOp2L
$Hidden_layer_2/MatMul/ReadVariableOp$Hidden_layer_2/MatMul/ReadVariableOp2r
7Hidden_layer_2/kernel/Regularizer/L2Loss/ReadVariableOp7Hidden_layer_2/kernel/Regularizer/L2Loss/ReadVariableOp2J
#output_layer/BiasAdd/ReadVariableOp#output_layer/BiasAdd/ReadVariableOp2H
"output_layer/MatMul/ReadVariableOp"output_layer/MatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�+
�
E__inference_sequential_layer_call_and_return_conditional_losses_34232

inputs&
hidden_layer_0_34155:"
hidden_layer_0_34157:&
hidden_layer_1_34176:"
hidden_layer_1_34178:&
hidden_layer_2_34197:"
hidden_layer_2_34199:$
output_layer_34214: 
output_layer_34216:
identity��&Hidden_layer_0/StatefulPartitionedCall�7Hidden_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp�&Hidden_layer_1/StatefulPartitionedCall�7Hidden_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp�&Hidden_layer_2/StatefulPartitionedCall�7Hidden_layer_2/kernel/Regularizer/L2Loss/ReadVariableOp�$output_layer/StatefulPartitionedCall�
&Hidden_layer_0/StatefulPartitionedCallStatefulPartitionedCallinputshidden_layer_0_34155hidden_layer_0_34157*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_Hidden_layer_0_layer_call_and_return_conditional_losses_34154�
&Hidden_layer_1/StatefulPartitionedCallStatefulPartitionedCall/Hidden_layer_0/StatefulPartitionedCall:output:0hidden_layer_1_34176hidden_layer_1_34178*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_Hidden_layer_1_layer_call_and_return_conditional_losses_34175�
&Hidden_layer_2/StatefulPartitionedCallStatefulPartitionedCall/Hidden_layer_1/StatefulPartitionedCall:output:0hidden_layer_2_34197hidden_layer_2_34199*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_Hidden_layer_2_layer_call_and_return_conditional_losses_34196�
$output_layer/StatefulPartitionedCallStatefulPartitionedCall/Hidden_layer_2/StatefulPartitionedCall:output:0output_layer_34214output_layer_34216*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_output_layer_layer_call_and_return_conditional_losses_34213�
7Hidden_layer_0/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOphidden_layer_0_34155*
_output_shapes

:*
dtype0�
(Hidden_layer_0/kernel/Regularizer/L2LossL2Loss?Hidden_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: l
'Hidden_layer_0/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
%Hidden_layer_0/kernel/Regularizer/mulMul0Hidden_layer_0/kernel/Regularizer/mul/x:output:01Hidden_layer_0/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
7Hidden_layer_1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOphidden_layer_1_34176*
_output_shapes

:*
dtype0�
(Hidden_layer_1/kernel/Regularizer/L2LossL2Loss?Hidden_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: l
'Hidden_layer_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
%Hidden_layer_1/kernel/Regularizer/mulMul0Hidden_layer_1/kernel/Regularizer/mul/x:output:01Hidden_layer_1/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
7Hidden_layer_2/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOphidden_layer_2_34197*
_output_shapes

:*
dtype0�
(Hidden_layer_2/kernel/Regularizer/L2LossL2Loss?Hidden_layer_2/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: l
'Hidden_layer_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
%Hidden_layer_2/kernel/Regularizer/mulMul0Hidden_layer_2/kernel/Regularizer/mul/x:output:01Hidden_layer_2/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: |
IdentityIdentity-output_layer/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp'^Hidden_layer_0/StatefulPartitionedCall8^Hidden_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp'^Hidden_layer_1/StatefulPartitionedCall8^Hidden_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp'^Hidden_layer_2/StatefulPartitionedCall8^Hidden_layer_2/kernel/Regularizer/L2Loss/ReadVariableOp%^output_layer/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : : : : : : : 2P
&Hidden_layer_0/StatefulPartitionedCall&Hidden_layer_0/StatefulPartitionedCall2r
7Hidden_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp7Hidden_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp2P
&Hidden_layer_1/StatefulPartitionedCall&Hidden_layer_1/StatefulPartitionedCall2r
7Hidden_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp7Hidden_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp2P
&Hidden_layer_2/StatefulPartitionedCall&Hidden_layer_2/StatefulPartitionedCall2r
7Hidden_layer_2/kernel/Regularizer/L2Loss/ReadVariableOp7Hidden_layer_2/kernel/Regularizer/L2Loss/ReadVariableOp2L
$output_layer/StatefulPartitionedCall$output_layer/StatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
.__inference_Hidden_layer_2_layer_call_fn_34702

inputs
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_Hidden_layer_2_layer_call_and_return_conditional_losses_34196o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�/
�
 __inference__wrapped_model_34132
input_layerJ
8sequential_hidden_layer_0_matmul_readvariableop_resource:G
9sequential_hidden_layer_0_biasadd_readvariableop_resource:J
8sequential_hidden_layer_1_matmul_readvariableop_resource:G
9sequential_hidden_layer_1_biasadd_readvariableop_resource:J
8sequential_hidden_layer_2_matmul_readvariableop_resource:G
9sequential_hidden_layer_2_biasadd_readvariableop_resource:H
6sequential_output_layer_matmul_readvariableop_resource:E
7sequential_output_layer_biasadd_readvariableop_resource:
identity��0sequential/Hidden_layer_0/BiasAdd/ReadVariableOp�/sequential/Hidden_layer_0/MatMul/ReadVariableOp�0sequential/Hidden_layer_1/BiasAdd/ReadVariableOp�/sequential/Hidden_layer_1/MatMul/ReadVariableOp�0sequential/Hidden_layer_2/BiasAdd/ReadVariableOp�/sequential/Hidden_layer_2/MatMul/ReadVariableOp�.sequential/output_layer/BiasAdd/ReadVariableOp�-sequential/output_layer/MatMul/ReadVariableOp�
/sequential/Hidden_layer_0/MatMul/ReadVariableOpReadVariableOp8sequential_hidden_layer_0_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
 sequential/Hidden_layer_0/MatMulMatMulinput_layer7sequential/Hidden_layer_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
0sequential/Hidden_layer_0/BiasAdd/ReadVariableOpReadVariableOp9sequential_hidden_layer_0_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
!sequential/Hidden_layer_0/BiasAddBiasAdd*sequential/Hidden_layer_0/MatMul:product:08sequential/Hidden_layer_0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
sequential/Hidden_layer_0/EluElu*sequential/Hidden_layer_0/BiasAdd:output:0*
T0*'
_output_shapes
:����������
/sequential/Hidden_layer_1/MatMul/ReadVariableOpReadVariableOp8sequential_hidden_layer_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
 sequential/Hidden_layer_1/MatMulMatMul+sequential/Hidden_layer_0/Elu:activations:07sequential/Hidden_layer_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
0sequential/Hidden_layer_1/BiasAdd/ReadVariableOpReadVariableOp9sequential_hidden_layer_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
!sequential/Hidden_layer_1/BiasAddBiasAdd*sequential/Hidden_layer_1/MatMul:product:08sequential/Hidden_layer_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
sequential/Hidden_layer_1/EluElu*sequential/Hidden_layer_1/BiasAdd:output:0*
T0*'
_output_shapes
:����������
/sequential/Hidden_layer_2/MatMul/ReadVariableOpReadVariableOp8sequential_hidden_layer_2_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
 sequential/Hidden_layer_2/MatMulMatMul+sequential/Hidden_layer_1/Elu:activations:07sequential/Hidden_layer_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
0sequential/Hidden_layer_2/BiasAdd/ReadVariableOpReadVariableOp9sequential_hidden_layer_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
!sequential/Hidden_layer_2/BiasAddBiasAdd*sequential/Hidden_layer_2/MatMul:product:08sequential/Hidden_layer_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
sequential/Hidden_layer_2/EluElu*sequential/Hidden_layer_2/BiasAdd:output:0*
T0*'
_output_shapes
:����������
-sequential/output_layer/MatMul/ReadVariableOpReadVariableOp6sequential_output_layer_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
sequential/output_layer/MatMulMatMul+sequential/Hidden_layer_2/Elu:activations:05sequential/output_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
.sequential/output_layer/BiasAdd/ReadVariableOpReadVariableOp7sequential_output_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
sequential/output_layer/BiasAddBiasAdd(sequential/output_layer/MatMul:product:06sequential/output_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������~
sequential/output_layer/EluElu(sequential/output_layer/BiasAdd:output:0*
T0*'
_output_shapes
:���������x
IdentityIdentity)sequential/output_layer/Elu:activations:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp1^sequential/Hidden_layer_0/BiasAdd/ReadVariableOp0^sequential/Hidden_layer_0/MatMul/ReadVariableOp1^sequential/Hidden_layer_1/BiasAdd/ReadVariableOp0^sequential/Hidden_layer_1/MatMul/ReadVariableOp1^sequential/Hidden_layer_2/BiasAdd/ReadVariableOp0^sequential/Hidden_layer_2/MatMul/ReadVariableOp/^sequential/output_layer/BiasAdd/ReadVariableOp.^sequential/output_layer/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : : : : : : : 2d
0sequential/Hidden_layer_0/BiasAdd/ReadVariableOp0sequential/Hidden_layer_0/BiasAdd/ReadVariableOp2b
/sequential/Hidden_layer_0/MatMul/ReadVariableOp/sequential/Hidden_layer_0/MatMul/ReadVariableOp2d
0sequential/Hidden_layer_1/BiasAdd/ReadVariableOp0sequential/Hidden_layer_1/BiasAdd/ReadVariableOp2b
/sequential/Hidden_layer_1/MatMul/ReadVariableOp/sequential/Hidden_layer_1/MatMul/ReadVariableOp2d
0sequential/Hidden_layer_2/BiasAdd/ReadVariableOp0sequential/Hidden_layer_2/BiasAdd/ReadVariableOp2b
/sequential/Hidden_layer_2/MatMul/ReadVariableOp/sequential/Hidden_layer_2/MatMul/ReadVariableOp2`
.sequential/output_layer/BiasAdd/ReadVariableOp.sequential/output_layer/BiasAdd/ReadVariableOp2^
-sequential/output_layer/MatMul/ReadVariableOp-sequential/output_layer/MatMul/ReadVariableOp:T P
'
_output_shapes
:���������
%
_user_specified_nameinput_layer
�J
�
__inference__traced_save_34892
file_prefix4
0savev2_hidden_layer_0_kernel_read_readvariableop2
.savev2_hidden_layer_0_bias_read_readvariableop4
0savev2_hidden_layer_1_kernel_read_readvariableop2
.savev2_hidden_layer_1_bias_read_readvariableop4
0savev2_hidden_layer_2_kernel_read_readvariableop2
.savev2_hidden_layer_2_bias_read_readvariableop2
.savev2_output_layer_kernel_read_readvariableop0
,savev2_output_layer_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop&
"savev2_total_2_read_readvariableop&
"savev2_count_2_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop;
7savev2_adam_hidden_layer_0_kernel_m_read_readvariableop9
5savev2_adam_hidden_layer_0_bias_m_read_readvariableop;
7savev2_adam_hidden_layer_1_kernel_m_read_readvariableop9
5savev2_adam_hidden_layer_1_bias_m_read_readvariableop;
7savev2_adam_hidden_layer_2_kernel_m_read_readvariableop9
5savev2_adam_hidden_layer_2_bias_m_read_readvariableop9
5savev2_adam_output_layer_kernel_m_read_readvariableop7
3savev2_adam_output_layer_bias_m_read_readvariableop;
7savev2_adam_hidden_layer_0_kernel_v_read_readvariableop9
5savev2_adam_hidden_layer_0_bias_v_read_readvariableop;
7savev2_adam_hidden_layer_1_kernel_v_read_readvariableop9
5savev2_adam_hidden_layer_1_bias_v_read_readvariableop;
7savev2_adam_hidden_layer_2_kernel_v_read_readvariableop9
5savev2_adam_hidden_layer_2_bias_v_read_readvariableop9
5savev2_adam_output_layer_kernel_v_read_readvariableop7
3savev2_adam_output_layer_bias_v_read_readvariableop
savev2_const

identity_1��MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: �
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:$*
dtype0*�
value�B�$B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:$*
dtype0*[
valueRBP$B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:00savev2_hidden_layer_0_kernel_read_readvariableop.savev2_hidden_layer_0_bias_read_readvariableop0savev2_hidden_layer_1_kernel_read_readvariableop.savev2_hidden_layer_1_bias_read_readvariableop0savev2_hidden_layer_2_kernel_read_readvariableop.savev2_hidden_layer_2_bias_read_readvariableop.savev2_output_layer_kernel_read_readvariableop,savev2_output_layer_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop"savev2_total_2_read_readvariableop"savev2_count_2_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop7savev2_adam_hidden_layer_0_kernel_m_read_readvariableop5savev2_adam_hidden_layer_0_bias_m_read_readvariableop7savev2_adam_hidden_layer_1_kernel_m_read_readvariableop5savev2_adam_hidden_layer_1_bias_m_read_readvariableop7savev2_adam_hidden_layer_2_kernel_m_read_readvariableop5savev2_adam_hidden_layer_2_bias_m_read_readvariableop5savev2_adam_output_layer_kernel_m_read_readvariableop3savev2_adam_output_layer_bias_m_read_readvariableop7savev2_adam_hidden_layer_0_kernel_v_read_readvariableop5savev2_adam_hidden_layer_0_bias_v_read_readvariableop7savev2_adam_hidden_layer_1_kernel_v_read_readvariableop5savev2_adam_hidden_layer_1_bias_v_read_readvariableop7savev2_adam_hidden_layer_2_kernel_v_read_readvariableop5savev2_adam_hidden_layer_2_bias_v_read_readvariableop5savev2_adam_output_layer_kernel_v_read_readvariableop3savev2_adam_output_layer_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *2
dtypes(
&2$	�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*�
_input_shapes�
�: ::::::::: : : : : : : : : : : ::::::::::::::::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::	

_output_shapes
: :


_output_shapes
: :
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
: :$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$  

_output_shapes

:: !

_output_shapes
::$" 

_output_shapes

:: #

_output_shapes
::$

_output_shapes
: 
�

�
G__inference_output_layer_layer_call_and_return_conditional_losses_34213

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������N
EluEluBiasAdd:output:0*
T0*'
_output_shapes
:���������`
IdentityIdentityElu:activations:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
I__inference_Hidden_layer_2_layer_call_and_return_conditional_losses_34196

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�7Hidden_layer_2/kernel/Regularizer/L2Loss/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������N
EluEluBiasAdd:output:0*
T0*'
_output_shapes
:����������
7Hidden_layer_2/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0�
(Hidden_layer_2/kernel/Regularizer/L2LossL2Loss?Hidden_layer_2/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: l
'Hidden_layer_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
%Hidden_layer_2/kernel/Regularizer/mulMul0Hidden_layer_2/kernel/Regularizer/mul/x:output:01Hidden_layer_2/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: `
IdentityIdentityElu:activations:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^BiasAdd/ReadVariableOp8^Hidden_layer_2/kernel/Regularizer/L2Loss/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2r
7Hidden_layer_2/kernel/Regularizer/L2Loss/ReadVariableOp7Hidden_layer_2/kernel/Regularizer/L2Loss/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�+
�
E__inference_sequential_layer_call_and_return_conditional_losses_34350

inputs&
hidden_layer_0_34317:"
hidden_layer_0_34319:&
hidden_layer_1_34322:"
hidden_layer_1_34324:&
hidden_layer_2_34327:"
hidden_layer_2_34329:$
output_layer_34332: 
output_layer_34334:
identity��&Hidden_layer_0/StatefulPartitionedCall�7Hidden_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp�&Hidden_layer_1/StatefulPartitionedCall�7Hidden_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp�&Hidden_layer_2/StatefulPartitionedCall�7Hidden_layer_2/kernel/Regularizer/L2Loss/ReadVariableOp�$output_layer/StatefulPartitionedCall�
&Hidden_layer_0/StatefulPartitionedCallStatefulPartitionedCallinputshidden_layer_0_34317hidden_layer_0_34319*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_Hidden_layer_0_layer_call_and_return_conditional_losses_34154�
&Hidden_layer_1/StatefulPartitionedCallStatefulPartitionedCall/Hidden_layer_0/StatefulPartitionedCall:output:0hidden_layer_1_34322hidden_layer_1_34324*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_Hidden_layer_1_layer_call_and_return_conditional_losses_34175�
&Hidden_layer_2/StatefulPartitionedCallStatefulPartitionedCall/Hidden_layer_1/StatefulPartitionedCall:output:0hidden_layer_2_34327hidden_layer_2_34329*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_Hidden_layer_2_layer_call_and_return_conditional_losses_34196�
$output_layer/StatefulPartitionedCallStatefulPartitionedCall/Hidden_layer_2/StatefulPartitionedCall:output:0output_layer_34332output_layer_34334*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_output_layer_layer_call_and_return_conditional_losses_34213�
7Hidden_layer_0/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOphidden_layer_0_34317*
_output_shapes

:*
dtype0�
(Hidden_layer_0/kernel/Regularizer/L2LossL2Loss?Hidden_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: l
'Hidden_layer_0/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
%Hidden_layer_0/kernel/Regularizer/mulMul0Hidden_layer_0/kernel/Regularizer/mul/x:output:01Hidden_layer_0/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
7Hidden_layer_1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOphidden_layer_1_34322*
_output_shapes

:*
dtype0�
(Hidden_layer_1/kernel/Regularizer/L2LossL2Loss?Hidden_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: l
'Hidden_layer_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
%Hidden_layer_1/kernel/Regularizer/mulMul0Hidden_layer_1/kernel/Regularizer/mul/x:output:01Hidden_layer_1/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
7Hidden_layer_2/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOphidden_layer_2_34327*
_output_shapes

:*
dtype0�
(Hidden_layer_2/kernel/Regularizer/L2LossL2Loss?Hidden_layer_2/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: l
'Hidden_layer_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
%Hidden_layer_2/kernel/Regularizer/mulMul0Hidden_layer_2/kernel/Regularizer/mul/x:output:01Hidden_layer_2/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: |
IdentityIdentity-output_layer/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp'^Hidden_layer_0/StatefulPartitionedCall8^Hidden_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp'^Hidden_layer_1/StatefulPartitionedCall8^Hidden_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp'^Hidden_layer_2/StatefulPartitionedCall8^Hidden_layer_2/kernel/Regularizer/L2Loss/ReadVariableOp%^output_layer/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : : : : : : : 2P
&Hidden_layer_0/StatefulPartitionedCall&Hidden_layer_0/StatefulPartitionedCall2r
7Hidden_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp7Hidden_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp2P
&Hidden_layer_1/StatefulPartitionedCall&Hidden_layer_1/StatefulPartitionedCall2r
7Hidden_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp7Hidden_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp2P
&Hidden_layer_2/StatefulPartitionedCall&Hidden_layer_2/StatefulPartitionedCall2r
7Hidden_layer_2/kernel/Regularizer/L2Loss/ReadVariableOp7Hidden_layer_2/kernel/Regularizer/L2Loss/ReadVariableOp2L
$output_layer/StatefulPartitionedCall$output_layer/StatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�+
�
E__inference_sequential_layer_call_and_return_conditional_losses_34462
input_layer&
hidden_layer_0_34429:"
hidden_layer_0_34431:&
hidden_layer_1_34434:"
hidden_layer_1_34436:&
hidden_layer_2_34439:"
hidden_layer_2_34441:$
output_layer_34444: 
output_layer_34446:
identity��&Hidden_layer_0/StatefulPartitionedCall�7Hidden_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp�&Hidden_layer_1/StatefulPartitionedCall�7Hidden_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp�&Hidden_layer_2/StatefulPartitionedCall�7Hidden_layer_2/kernel/Regularizer/L2Loss/ReadVariableOp�$output_layer/StatefulPartitionedCall�
&Hidden_layer_0/StatefulPartitionedCallStatefulPartitionedCallinput_layerhidden_layer_0_34429hidden_layer_0_34431*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_Hidden_layer_0_layer_call_and_return_conditional_losses_34154�
&Hidden_layer_1/StatefulPartitionedCallStatefulPartitionedCall/Hidden_layer_0/StatefulPartitionedCall:output:0hidden_layer_1_34434hidden_layer_1_34436*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_Hidden_layer_1_layer_call_and_return_conditional_losses_34175�
&Hidden_layer_2/StatefulPartitionedCallStatefulPartitionedCall/Hidden_layer_1/StatefulPartitionedCall:output:0hidden_layer_2_34439hidden_layer_2_34441*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_Hidden_layer_2_layer_call_and_return_conditional_losses_34196�
$output_layer/StatefulPartitionedCallStatefulPartitionedCall/Hidden_layer_2/StatefulPartitionedCall:output:0output_layer_34444output_layer_34446*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_output_layer_layer_call_and_return_conditional_losses_34213�
7Hidden_layer_0/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOphidden_layer_0_34429*
_output_shapes

:*
dtype0�
(Hidden_layer_0/kernel/Regularizer/L2LossL2Loss?Hidden_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: l
'Hidden_layer_0/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
%Hidden_layer_0/kernel/Regularizer/mulMul0Hidden_layer_0/kernel/Regularizer/mul/x:output:01Hidden_layer_0/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
7Hidden_layer_1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOphidden_layer_1_34434*
_output_shapes

:*
dtype0�
(Hidden_layer_1/kernel/Regularizer/L2LossL2Loss?Hidden_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: l
'Hidden_layer_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
%Hidden_layer_1/kernel/Regularizer/mulMul0Hidden_layer_1/kernel/Regularizer/mul/x:output:01Hidden_layer_1/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
7Hidden_layer_2/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOphidden_layer_2_34439*
_output_shapes

:*
dtype0�
(Hidden_layer_2/kernel/Regularizer/L2LossL2Loss?Hidden_layer_2/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: l
'Hidden_layer_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
%Hidden_layer_2/kernel/Regularizer/mulMul0Hidden_layer_2/kernel/Regularizer/mul/x:output:01Hidden_layer_2/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: |
IdentityIdentity-output_layer/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp'^Hidden_layer_0/StatefulPartitionedCall8^Hidden_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp'^Hidden_layer_1/StatefulPartitionedCall8^Hidden_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp'^Hidden_layer_2/StatefulPartitionedCall8^Hidden_layer_2/kernel/Regularizer/L2Loss/ReadVariableOp%^output_layer/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : : : : : : : 2P
&Hidden_layer_0/StatefulPartitionedCall&Hidden_layer_0/StatefulPartitionedCall2r
7Hidden_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp7Hidden_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp2P
&Hidden_layer_1/StatefulPartitionedCall&Hidden_layer_1/StatefulPartitionedCall2r
7Hidden_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp7Hidden_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp2P
&Hidden_layer_2/StatefulPartitionedCall&Hidden_layer_2/StatefulPartitionedCall2r
7Hidden_layer_2/kernel/Regularizer/L2Loss/ReadVariableOp7Hidden_layer_2/kernel/Regularizer/L2Loss/ReadVariableOp2L
$output_layer/StatefulPartitionedCall$output_layer/StatefulPartitionedCall:T P
'
_output_shapes
:���������
%
_user_specified_nameinput_layer
�
�
.__inference_Hidden_layer_0_layer_call_fn_34654

inputs
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_Hidden_layer_0_layer_call_and_return_conditional_losses_34154o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
I__inference_Hidden_layer_0_layer_call_and_return_conditional_losses_34669

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�7Hidden_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������N
EluEluBiasAdd:output:0*
T0*'
_output_shapes
:����������
7Hidden_layer_0/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0�
(Hidden_layer_0/kernel/Regularizer/L2LossL2Loss?Hidden_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: l
'Hidden_layer_0/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
%Hidden_layer_0/kernel/Regularizer/mulMul0Hidden_layer_0/kernel/Regularizer/mul/x:output:01Hidden_layer_0/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: `
IdentityIdentityElu:activations:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^BiasAdd/ReadVariableOp8^Hidden_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2r
7Hidden_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp7Hidden_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
��
�
!__inference__traced_restore_35007
file_prefix8
&assignvariableop_hidden_layer_0_kernel:4
&assignvariableop_1_hidden_layer_0_bias::
(assignvariableop_2_hidden_layer_1_kernel:4
&assignvariableop_3_hidden_layer_1_bias::
(assignvariableop_4_hidden_layer_2_kernel:4
&assignvariableop_5_hidden_layer_2_bias:8
&assignvariableop_6_output_layer_kernel:2
$assignvariableop_7_output_layer_bias:&
assignvariableop_8_adam_iter:	 (
assignvariableop_9_adam_beta_1: )
assignvariableop_10_adam_beta_2: (
assignvariableop_11_adam_decay: 0
&assignvariableop_12_adam_learning_rate: %
assignvariableop_13_total_2: %
assignvariableop_14_count_2: %
assignvariableop_15_total_1: %
assignvariableop_16_count_1: #
assignvariableop_17_total: #
assignvariableop_18_count: B
0assignvariableop_19_adam_hidden_layer_0_kernel_m:<
.assignvariableop_20_adam_hidden_layer_0_bias_m:B
0assignvariableop_21_adam_hidden_layer_1_kernel_m:<
.assignvariableop_22_adam_hidden_layer_1_bias_m:B
0assignvariableop_23_adam_hidden_layer_2_kernel_m:<
.assignvariableop_24_adam_hidden_layer_2_bias_m:@
.assignvariableop_25_adam_output_layer_kernel_m::
,assignvariableop_26_adam_output_layer_bias_m:B
0assignvariableop_27_adam_hidden_layer_0_kernel_v:<
.assignvariableop_28_adam_hidden_layer_0_bias_v:B
0assignvariableop_29_adam_hidden_layer_1_kernel_v:<
.assignvariableop_30_adam_hidden_layer_1_bias_v:B
0assignvariableop_31_adam_hidden_layer_2_kernel_v:<
.assignvariableop_32_adam_hidden_layer_2_bias_v:@
.assignvariableop_33_adam_output_layer_kernel_v::
,assignvariableop_34_adam_output_layer_bias_v:
identity_36��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_34�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:$*
dtype0*�
value�B�$B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:$*
dtype0*[
valueRBP$B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::*2
dtypes(
&2$	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOp&assignvariableop_hidden_layer_0_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOp&assignvariableop_1_hidden_layer_0_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp(assignvariableop_2_hidden_layer_1_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOp&assignvariableop_3_hidden_layer_1_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp(assignvariableop_4_hidden_layer_2_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOp&assignvariableop_5_hidden_layer_2_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp&assignvariableop_6_output_layer_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOp$assignvariableop_7_output_layer_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_iterIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOpassignvariableop_9_adam_beta_1Identity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOpassignvariableop_10_adam_beta_2Identity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOpassignvariableop_11_adam_decayIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOp&assignvariableop_12_adam_learning_rateIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOpassignvariableop_13_total_2Identity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOpassignvariableop_14_count_2Identity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOpassignvariableop_15_total_1Identity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOpassignvariableop_16_count_1Identity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOpassignvariableop_17_totalIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOpassignvariableop_18_countIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOp0assignvariableop_19_adam_hidden_layer_0_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOp.assignvariableop_20_adam_hidden_layer_0_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOp0assignvariableop_21_adam_hidden_layer_1_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_22AssignVariableOp.assignvariableop_22_adam_hidden_layer_1_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_23AssignVariableOp0assignvariableop_23_adam_hidden_layer_2_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_24AssignVariableOp.assignvariableop_24_adam_hidden_layer_2_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_25AssignVariableOp.assignvariableop_25_adam_output_layer_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_26AssignVariableOp,assignvariableop_26_adam_output_layer_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_27AssignVariableOp0assignvariableop_27_adam_hidden_layer_0_kernel_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_28AssignVariableOp.assignvariableop_28_adam_hidden_layer_0_bias_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_29AssignVariableOp0assignvariableop_29_adam_hidden_layer_1_kernel_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_30AssignVariableOp.assignvariableop_30_adam_hidden_layer_1_bias_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_31AssignVariableOp0assignvariableop_31_adam_hidden_layer_2_kernel_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_32AssignVariableOp.assignvariableop_32_adam_hidden_layer_2_bias_vIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_33AssignVariableOp.assignvariableop_33_adam_output_layer_kernel_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_34AssignVariableOp,assignvariableop_34_adam_output_layer_bias_vIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 �
Identity_35Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_36IdentityIdentity_35:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_36Identity_36:output:0*[
_input_shapesJ
H: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_34AssignVariableOp_342(
AssignVariableOp_4AssignVariableOp_42(
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
�	
�
*__inference_sequential_layer_call_fn_34251
input_layer
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_layerunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_34232o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
'
_output_shapes
:���������
%
_user_specified_nameinput_layer
�
�
I__inference_Hidden_layer_2_layer_call_and_return_conditional_losses_34717

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�7Hidden_layer_2/kernel/Regularizer/L2Loss/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������N
EluEluBiasAdd:output:0*
T0*'
_output_shapes
:����������
7Hidden_layer_2/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0�
(Hidden_layer_2/kernel/Regularizer/L2LossL2Loss?Hidden_layer_2/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: l
'Hidden_layer_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
%Hidden_layer_2/kernel/Regularizer/mulMul0Hidden_layer_2/kernel/Regularizer/mul/x:output:01Hidden_layer_2/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: `
IdentityIdentityElu:activations:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^BiasAdd/ReadVariableOp8^Hidden_layer_2/kernel/Regularizer/L2Loss/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2r
7Hidden_layer_2/kernel/Regularizer/L2Loss/ReadVariableOp7Hidden_layer_2/kernel/Regularizer/L2Loss/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�;
�
E__inference_sequential_layer_call_and_return_conditional_losses_34645

inputs?
-hidden_layer_0_matmul_readvariableop_resource:<
.hidden_layer_0_biasadd_readvariableop_resource:?
-hidden_layer_1_matmul_readvariableop_resource:<
.hidden_layer_1_biasadd_readvariableop_resource:?
-hidden_layer_2_matmul_readvariableop_resource:<
.hidden_layer_2_biasadd_readvariableop_resource:=
+output_layer_matmul_readvariableop_resource::
,output_layer_biasadd_readvariableop_resource:
identity��%Hidden_layer_0/BiasAdd/ReadVariableOp�$Hidden_layer_0/MatMul/ReadVariableOp�7Hidden_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp�%Hidden_layer_1/BiasAdd/ReadVariableOp�$Hidden_layer_1/MatMul/ReadVariableOp�7Hidden_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp�%Hidden_layer_2/BiasAdd/ReadVariableOp�$Hidden_layer_2/MatMul/ReadVariableOp�7Hidden_layer_2/kernel/Regularizer/L2Loss/ReadVariableOp�#output_layer/BiasAdd/ReadVariableOp�"output_layer/MatMul/ReadVariableOp�
$Hidden_layer_0/MatMul/ReadVariableOpReadVariableOp-hidden_layer_0_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
Hidden_layer_0/MatMulMatMulinputs,Hidden_layer_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
%Hidden_layer_0/BiasAdd/ReadVariableOpReadVariableOp.hidden_layer_0_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
Hidden_layer_0/BiasAddBiasAddHidden_layer_0/MatMul:product:0-Hidden_layer_0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������l
Hidden_layer_0/EluEluHidden_layer_0/BiasAdd:output:0*
T0*'
_output_shapes
:����������
$Hidden_layer_1/MatMul/ReadVariableOpReadVariableOp-hidden_layer_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
Hidden_layer_1/MatMulMatMul Hidden_layer_0/Elu:activations:0,Hidden_layer_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
%Hidden_layer_1/BiasAdd/ReadVariableOpReadVariableOp.hidden_layer_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
Hidden_layer_1/BiasAddBiasAddHidden_layer_1/MatMul:product:0-Hidden_layer_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������l
Hidden_layer_1/EluEluHidden_layer_1/BiasAdd:output:0*
T0*'
_output_shapes
:����������
$Hidden_layer_2/MatMul/ReadVariableOpReadVariableOp-hidden_layer_2_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
Hidden_layer_2/MatMulMatMul Hidden_layer_1/Elu:activations:0,Hidden_layer_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
%Hidden_layer_2/BiasAdd/ReadVariableOpReadVariableOp.hidden_layer_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
Hidden_layer_2/BiasAddBiasAddHidden_layer_2/MatMul:product:0-Hidden_layer_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������l
Hidden_layer_2/EluEluHidden_layer_2/BiasAdd:output:0*
T0*'
_output_shapes
:����������
"output_layer/MatMul/ReadVariableOpReadVariableOp+output_layer_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
output_layer/MatMulMatMul Hidden_layer_2/Elu:activations:0*output_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
#output_layer/BiasAdd/ReadVariableOpReadVariableOp,output_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
output_layer/BiasAddBiasAddoutput_layer/MatMul:product:0+output_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������h
output_layer/EluEluoutput_layer/BiasAdd:output:0*
T0*'
_output_shapes
:����������
7Hidden_layer_0/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp-hidden_layer_0_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
(Hidden_layer_0/kernel/Regularizer/L2LossL2Loss?Hidden_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: l
'Hidden_layer_0/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
%Hidden_layer_0/kernel/Regularizer/mulMul0Hidden_layer_0/kernel/Regularizer/mul/x:output:01Hidden_layer_0/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
7Hidden_layer_1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp-hidden_layer_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
(Hidden_layer_1/kernel/Regularizer/L2LossL2Loss?Hidden_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: l
'Hidden_layer_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
%Hidden_layer_1/kernel/Regularizer/mulMul0Hidden_layer_1/kernel/Regularizer/mul/x:output:01Hidden_layer_1/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
7Hidden_layer_2/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp-hidden_layer_2_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
(Hidden_layer_2/kernel/Regularizer/L2LossL2Loss?Hidden_layer_2/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: l
'Hidden_layer_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
%Hidden_layer_2/kernel/Regularizer/mulMul0Hidden_layer_2/kernel/Regularizer/mul/x:output:01Hidden_layer_2/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: m
IdentityIdentityoutput_layer/Elu:activations:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp&^Hidden_layer_0/BiasAdd/ReadVariableOp%^Hidden_layer_0/MatMul/ReadVariableOp8^Hidden_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp&^Hidden_layer_1/BiasAdd/ReadVariableOp%^Hidden_layer_1/MatMul/ReadVariableOp8^Hidden_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp&^Hidden_layer_2/BiasAdd/ReadVariableOp%^Hidden_layer_2/MatMul/ReadVariableOp8^Hidden_layer_2/kernel/Regularizer/L2Loss/ReadVariableOp$^output_layer/BiasAdd/ReadVariableOp#^output_layer/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : : : : : : : 2N
%Hidden_layer_0/BiasAdd/ReadVariableOp%Hidden_layer_0/BiasAdd/ReadVariableOp2L
$Hidden_layer_0/MatMul/ReadVariableOp$Hidden_layer_0/MatMul/ReadVariableOp2r
7Hidden_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp7Hidden_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp2N
%Hidden_layer_1/BiasAdd/ReadVariableOp%Hidden_layer_1/BiasAdd/ReadVariableOp2L
$Hidden_layer_1/MatMul/ReadVariableOp$Hidden_layer_1/MatMul/ReadVariableOp2r
7Hidden_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp7Hidden_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp2N
%Hidden_layer_2/BiasAdd/ReadVariableOp%Hidden_layer_2/BiasAdd/ReadVariableOp2L
$Hidden_layer_2/MatMul/ReadVariableOp$Hidden_layer_2/MatMul/ReadVariableOp2r
7Hidden_layer_2/kernel/Regularizer/L2Loss/ReadVariableOp7Hidden_layer_2/kernel/Regularizer/L2Loss/ReadVariableOp2J
#output_layer/BiasAdd/ReadVariableOp#output_layer/BiasAdd/ReadVariableOp2H
"output_layer/MatMul/ReadVariableOp"output_layer/MatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�	
�
*__inference_sequential_layer_call_fn_34557

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_34350o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�	
�
*__inference_sequential_layer_call_fn_34536

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_34232o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs"�	L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
C
input_layer4
serving_default_input_layer:0���������@
output_layer0
StatefulPartitionedCall:0���������tensorflow/serving/predict:ɏ
�
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
	variables
trainable_variables
regularization_losses
	keras_api
	__call__
*
&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures"
_tf_keras_sequential
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
�
	variables
trainable_variables
 regularization_losses
!	keras_api
"__call__
*#&call_and_return_all_conditional_losses

$kernel
%bias"
_tf_keras_layer
�
&	variables
'trainable_variables
(regularization_losses
)	keras_api
*__call__
*+&call_and_return_all_conditional_losses

,kernel
-bias"
_tf_keras_layer
X
0
1
2
3
$4
%5
,6
-7"
trackable_list_wrapper
X
0
1
2
3
$4
%5
,6
-7"
trackable_list_wrapper
5
.0
/1
02"
trackable_list_wrapper
�
1non_trainable_variables

2layers
3metrics
4layer_regularization_losses
5layer_metrics
	variables
trainable_variables
regularization_losses
	__call__
_default_save_signature
*
&call_and_return_all_conditional_losses
&
"call_and_return_conditional_losses"
_generic_user_object
�
6trace_0
7trace_1
8trace_2
9trace_32�
*__inference_sequential_layer_call_fn_34251
*__inference_sequential_layer_call_fn_34536
*__inference_sequential_layer_call_fn_34557
*__inference_sequential_layer_call_fn_34390�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z6trace_0z7trace_1z8trace_2z9trace_3
�
:trace_0
;trace_1
<trace_2
=trace_32�
E__inference_sequential_layer_call_and_return_conditional_losses_34601
E__inference_sequential_layer_call_and_return_conditional_losses_34645
E__inference_sequential_layer_call_and_return_conditional_losses_34426
E__inference_sequential_layer_call_and_return_conditional_losses_34462�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z:trace_0z;trace_1z<trace_2z=trace_3
�B�
 __inference__wrapped_model_34132input_layer"�
���
FullArgSpec
args� 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�
>iter

?beta_1

@beta_2
	Adecay
Blearning_ratemtmumvmw$mx%my,mz-m{v|v}v~v$v�%v�,v�-v�"
	optimizer
,
Cserving_default"
signature_map
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
'
.0"
trackable_list_wrapper
�
Dnon_trainable_variables

Elayers
Fmetrics
Glayer_regularization_losses
Hlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
Itrace_02�
.__inference_Hidden_layer_0_layer_call_fn_34654�
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
 zItrace_0
�
Jtrace_02�
I__inference_Hidden_layer_0_layer_call_and_return_conditional_losses_34669�
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
 zJtrace_0
':%2Hidden_layer_0/kernel
!:2Hidden_layer_0/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
'
/0"
trackable_list_wrapper
�
Knon_trainable_variables

Llayers
Mmetrics
Nlayer_regularization_losses
Olayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
Ptrace_02�
.__inference_Hidden_layer_1_layer_call_fn_34678�
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
 zPtrace_0
�
Qtrace_02�
I__inference_Hidden_layer_1_layer_call_and_return_conditional_losses_34693�
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
 zQtrace_0
':%2Hidden_layer_1/kernel
!:2Hidden_layer_1/bias
.
$0
%1"
trackable_list_wrapper
.
$0
%1"
trackable_list_wrapper
'
00"
trackable_list_wrapper
�
Rnon_trainable_variables

Slayers
Tmetrics
Ulayer_regularization_losses
Vlayer_metrics
	variables
trainable_variables
 regularization_losses
"__call__
*#&call_and_return_all_conditional_losses
&#"call_and_return_conditional_losses"
_generic_user_object
�
Wtrace_02�
.__inference_Hidden_layer_2_layer_call_fn_34702�
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
 zWtrace_0
�
Xtrace_02�
I__inference_Hidden_layer_2_layer_call_and_return_conditional_losses_34717�
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
 zXtrace_0
':%2Hidden_layer_2/kernel
!:2Hidden_layer_2/bias
.
,0
-1"
trackable_list_wrapper
.
,0
-1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Ynon_trainable_variables

Zlayers
[metrics
\layer_regularization_losses
]layer_metrics
&	variables
'trainable_variables
(regularization_losses
*__call__
*+&call_and_return_all_conditional_losses
&+"call_and_return_conditional_losses"
_generic_user_object
�
^trace_02�
,__inference_output_layer_layer_call_fn_34726�
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
 z^trace_0
�
_trace_02�
G__inference_output_layer_layer_call_and_return_conditional_losses_34737�
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
 z_trace_0
%:#2output_layer/kernel
:2output_layer/bias
�
`trace_02�
__inference_loss_fn_0_34746�
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
annotations� *� z`trace_0
�
atrace_02�
__inference_loss_fn_1_34755�
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
annotations� *� zatrace_0
�
btrace_02�
__inference_loss_fn_2_34764�
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
annotations� *� zbtrace_0
 "
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
5
c0
d1
e2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
*__inference_sequential_layer_call_fn_34251input_layer"�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
*__inference_sequential_layer_call_fn_34536inputs"�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
*__inference_sequential_layer_call_fn_34557inputs"�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
*__inference_sequential_layer_call_fn_34390input_layer"�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_sequential_layer_call_and_return_conditional_losses_34601inputs"�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_sequential_layer_call_and_return_conditional_losses_34645inputs"�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_sequential_layer_call_and_return_conditional_losses_34426input_layer"�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_sequential_layer_call_and_return_conditional_losses_34462input_layer"�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
�B�
#__inference_signature_wrapper_34503input_layer"�
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
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
.0"
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
.__inference_Hidden_layer_0_layer_call_fn_34654inputs"�
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
�B�
I__inference_Hidden_layer_0_layer_call_and_return_conditional_losses_34669inputs"�
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
/0"
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
.__inference_Hidden_layer_1_layer_call_fn_34678inputs"�
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
�B�
I__inference_Hidden_layer_1_layer_call_and_return_conditional_losses_34693inputs"�
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
00"
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
.__inference_Hidden_layer_2_layer_call_fn_34702inputs"�
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
�B�
I__inference_Hidden_layer_2_layer_call_and_return_conditional_losses_34717inputs"�
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
�B�
,__inference_output_layer_layer_call_fn_34726inputs"�
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
�B�
G__inference_output_layer_layer_call_and_return_conditional_losses_34737inputs"�
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
�B�
__inference_loss_fn_0_34746"�
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
__inference_loss_fn_1_34755"�
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
__inference_loss_fn_2_34764"�
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
N
f	variables
g	keras_api
	htotal
	icount"
_tf_keras_metric
^
j	variables
k	keras_api
	ltotal
	mcount
n
_fn_kwargs"
_tf_keras_metric
^
o	variables
p	keras_api
	qtotal
	rcount
s
_fn_kwargs"
_tf_keras_metric
.
h0
i1"
trackable_list_wrapper
-
f	variables"
_generic_user_object
:  (2total
:  (2count
.
l0
m1"
trackable_list_wrapper
-
j	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
q0
r1"
trackable_list_wrapper
-
o	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
,:*2Adam/Hidden_layer_0/kernel/m
&:$2Adam/Hidden_layer_0/bias/m
,:*2Adam/Hidden_layer_1/kernel/m
&:$2Adam/Hidden_layer_1/bias/m
,:*2Adam/Hidden_layer_2/kernel/m
&:$2Adam/Hidden_layer_2/bias/m
*:(2Adam/output_layer/kernel/m
$:"2Adam/output_layer/bias/m
,:*2Adam/Hidden_layer_0/kernel/v
&:$2Adam/Hidden_layer_0/bias/v
,:*2Adam/Hidden_layer_1/kernel/v
&:$2Adam/Hidden_layer_1/bias/v
,:*2Adam/Hidden_layer_2/kernel/v
&:$2Adam/Hidden_layer_2/bias/v
*:(2Adam/output_layer/kernel/v
$:"2Adam/output_layer/bias/v�
I__inference_Hidden_layer_0_layer_call_and_return_conditional_losses_34669\/�,
%�"
 �
inputs���������
� "%�"
�
0���������
� �
.__inference_Hidden_layer_0_layer_call_fn_34654O/�,
%�"
 �
inputs���������
� "�����������
I__inference_Hidden_layer_1_layer_call_and_return_conditional_losses_34693\/�,
%�"
 �
inputs���������
� "%�"
�
0���������
� �
.__inference_Hidden_layer_1_layer_call_fn_34678O/�,
%�"
 �
inputs���������
� "�����������
I__inference_Hidden_layer_2_layer_call_and_return_conditional_losses_34717\$%/�,
%�"
 �
inputs���������
� "%�"
�
0���������
� �
.__inference_Hidden_layer_2_layer_call_fn_34702O$%/�,
%�"
 �
inputs���������
� "�����������
 __inference__wrapped_model_34132}$%,-4�1
*�'
%�"
input_layer���������
� ";�8
6
output_layer&�#
output_layer���������:
__inference_loss_fn_0_34746�

� 
� "� :
__inference_loss_fn_1_34755�

� 
� "� :
__inference_loss_fn_2_34764$�

� 
� "� �
G__inference_output_layer_layer_call_and_return_conditional_losses_34737\,-/�,
%�"
 �
inputs���������
� "%�"
�
0���������
� 
,__inference_output_layer_layer_call_fn_34726O,-/�,
%�"
 �
inputs���������
� "�����������
E__inference_sequential_layer_call_and_return_conditional_losses_34426o$%,-<�9
2�/
%�"
input_layer���������
p 

 
� "%�"
�
0���������
� �
E__inference_sequential_layer_call_and_return_conditional_losses_34462o$%,-<�9
2�/
%�"
input_layer���������
p

 
� "%�"
�
0���������
� �
E__inference_sequential_layer_call_and_return_conditional_losses_34601j$%,-7�4
-�*
 �
inputs���������
p 

 
� "%�"
�
0���������
� �
E__inference_sequential_layer_call_and_return_conditional_losses_34645j$%,-7�4
-�*
 �
inputs���������
p

 
� "%�"
�
0���������
� �
*__inference_sequential_layer_call_fn_34251b$%,-<�9
2�/
%�"
input_layer���������
p 

 
� "�����������
*__inference_sequential_layer_call_fn_34390b$%,-<�9
2�/
%�"
input_layer���������
p

 
� "�����������
*__inference_sequential_layer_call_fn_34536]$%,-7�4
-�*
 �
inputs���������
p 

 
� "�����������
*__inference_sequential_layer_call_fn_34557]$%,-7�4
-�*
 �
inputs���������
p

 
� "�����������
#__inference_signature_wrapper_34503�$%,-C�@
� 
9�6
4
input_layer%�"
input_layer���������";�8
6
output_layer&�#
output_layer���������