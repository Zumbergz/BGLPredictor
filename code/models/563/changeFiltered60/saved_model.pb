ео7
Ђл
D
AddV2
x"T
y"T
z"T"
Ttype:
2	ђљ
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( ѕ
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
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
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
є
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( ѕ
?
Mul
x"T
y"T
z"T"
Ttype:
2	љ

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
│
PartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetypeѕ
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0ѕ
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0ѕ
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
[
Split
	split_dim

value"T
output"T*	num_split"
	num_splitint(0"	
Ttype
┴
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
executor_typestring ѕе
џ
StatelessWhile

input2T
output2T"
T
list(type)("
condfunc"
bodyfunc" 
output_shapeslist(shape)
 "
parallel_iterationsint

@
StaticRegexFullMatch	
input

output
"
patternstring
э
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
-
Tanh
x"T
y"T"
Ttype:

2
░
TensorListFromTensor
tensor"element_dtype
element_shape"
shape_type/
output_handleіжУelement_dtype"
element_dtypetype"

shape_typetype:
2	
Ъ
TensorListReserve
element_shape"
shape_type
num_elements(
handleіжУelement_dtype"
element_dtypetype"

shape_typetype:
2	
ѕ
TensorListStack
input_handle
element_shape
tensor"element_dtype"
element_dtypetype" 
num_elementsint         
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
ќ
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ѕ"serve*2.11.02v2.11.0-rc2-17-gd5b57ca93e58ё№5
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
ђ
Adam/v/dense_13/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/v/dense_13/bias
y
(Adam/v/dense_13/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_13/bias*
_output_shapes
:*
dtype0
ђ
Adam/m/dense_13/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/m/dense_13/bias
y
(Adam/m/dense_13/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_13/bias*
_output_shapes
:*
dtype0
Ѕ
Adam/v/dense_13/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	а*'
shared_nameAdam/v/dense_13/kernel
ѓ
*Adam/v/dense_13/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_13/kernel*
_output_shapes
:	а*
dtype0
Ѕ
Adam/m/dense_13/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	а*'
shared_nameAdam/m/dense_13/kernel
ѓ
*Adam/m/dense_13/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_13/kernel*
_output_shapes
:	а*
dtype0
Ў
 Adam/v/lstm_13/lstm_cell_13/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ	*1
shared_name" Adam/v/lstm_13/lstm_cell_13/bias
њ
4Adam/v/lstm_13/lstm_cell_13/bias/Read/ReadVariableOpReadVariableOp Adam/v/lstm_13/lstm_cell_13/bias*
_output_shapes	
:ђ	*
dtype0
Ў
 Adam/m/lstm_13/lstm_cell_13/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ	*1
shared_name" Adam/m/lstm_13/lstm_cell_13/bias
њ
4Adam/m/lstm_13/lstm_cell_13/bias/Read/ReadVariableOpReadVariableOp Adam/m/lstm_13/lstm_cell_13/bias*
_output_shapes	
:ђ	*
dtype0
Х
,Adam/v/lstm_13/lstm_cell_13/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ађ	*=
shared_name.,Adam/v/lstm_13/lstm_cell_13/recurrent_kernel
»
@Adam/v/lstm_13/lstm_cell_13/recurrent_kernel/Read/ReadVariableOpReadVariableOp,Adam/v/lstm_13/lstm_cell_13/recurrent_kernel* 
_output_shapes
:
ађ	*
dtype0
Х
,Adam/m/lstm_13/lstm_cell_13/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ађ	*=
shared_name.,Adam/m/lstm_13/lstm_cell_13/recurrent_kernel
»
@Adam/m/lstm_13/lstm_cell_13/recurrent_kernel/Read/ReadVariableOpReadVariableOp,Adam/m/lstm_13/lstm_cell_13/recurrent_kernel* 
_output_shapes
:
ађ	*
dtype0
А
"Adam/v/lstm_13/lstm_cell_13/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ђ	*3
shared_name$"Adam/v/lstm_13/lstm_cell_13/kernel
џ
6Adam/v/lstm_13/lstm_cell_13/kernel/Read/ReadVariableOpReadVariableOp"Adam/v/lstm_13/lstm_cell_13/kernel*
_output_shapes
:	ђ	*
dtype0
А
"Adam/m/lstm_13/lstm_cell_13/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ђ	*3
shared_name$"Adam/m/lstm_13/lstm_cell_13/kernel
џ
6Adam/m/lstm_13/lstm_cell_13/kernel/Read/ReadVariableOpReadVariableOp"Adam/m/lstm_13/lstm_cell_13/kernel*
_output_shapes
:	ђ	*
dtype0
n
learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namelearning_rate
g
!learning_rate/Read/ReadVariableOpReadVariableOplearning_rate*
_output_shapes
: *
dtype0
f
	iterationVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	iteration
_
iteration/Read/ReadVariableOpReadVariableOp	iteration*
_output_shapes
: *
dtype0	
r
dense_13/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_13/bias
k
!dense_13/bias/Read/ReadVariableOpReadVariableOpdense_13/bias*
_output_shapes
:*
dtype0
{
dense_13/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	а* 
shared_namedense_13/kernel
t
#dense_13/kernel/Read/ReadVariableOpReadVariableOpdense_13/kernel*
_output_shapes
:	а*
dtype0
І
lstm_13/lstm_cell_13/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ	**
shared_namelstm_13/lstm_cell_13/bias
ё
-lstm_13/lstm_cell_13/bias/Read/ReadVariableOpReadVariableOplstm_13/lstm_cell_13/bias*
_output_shapes	
:ђ	*
dtype0
е
%lstm_13/lstm_cell_13/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ађ	*6
shared_name'%lstm_13/lstm_cell_13/recurrent_kernel
А
9lstm_13/lstm_cell_13/recurrent_kernel/Read/ReadVariableOpReadVariableOp%lstm_13/lstm_cell_13/recurrent_kernel* 
_output_shapes
:
ађ	*
dtype0
Њ
lstm_13/lstm_cell_13/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ђ	*,
shared_namelstm_13/lstm_cell_13/kernel
ї
/lstm_13/lstm_cell_13/kernel/Read/ReadVariableOpReadVariableOplstm_13/lstm_cell_13/kernel*
_output_shapes
:	ђ	*
dtype0
ѓ
serving_default_input_1Placeholder*+
_output_shapes
:         *
dtype0* 
shape:         
└
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1lstm_13/lstm_cell_13/kernel%lstm_13/lstm_cell_13/recurrent_kernellstm_13/lstm_cell_13/biasdense_13/kerneldense_13/bias*
Tin

2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         *'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8ѓ *.
f)R'
%__inference_signature_wrapper_1210250

NoOpNoOp
г6
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*у5
valueП5B┌5 BМ5
О
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	model
		optimizer


signatures*
'
0
1
2
3
4*
'
0
1
2
3
4*
* 
░
non_trainable_variables

layers
metrics
layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
trace_0
trace_1
trace_2
trace_3* 
6
trace_0
trace_1
trace_2
trace_3* 
* 
в
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
 	variables
!trainable_variables
"regularization_losses
#	keras_api
$__call__
*%&call_and_return_all_conditional_losses*
Ђ
&
_variables
'_iterations
(_learning_rate
)_index_dict
*
_momentums
+_velocities
,_update_step_xla*

-serving_default* 
[U
VARIABLE_VALUElstm_13/lstm_cell_13/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUE%lstm_13/lstm_cell_13/recurrent_kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUElstm_13/lstm_cell_13/bias&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUEdense_13/kernel&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEdense_13/bias&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
* 

0*

.0
/1
02*
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
┴
1	variables
2trainable_variables
3regularization_losses
4	keras_api
5__call__
*6&call_and_return_all_conditional_losses
7_random_generator
8cell
9
state_spec*
д
:	variables
;trainable_variables
<regularization_losses
=	keras_api
>__call__
*?&call_and_return_all_conditional_losses

kernel
bias*
ј
@	variables
Atrainable_variables
Bregularization_losses
C	keras_api
D__call__
*E&call_and_return_all_conditional_losses* 
'
0
1
2
3
4*
'
0
1
2
3
4*
* 
Њ
Fnon_trainable_variables

Glayers
Hmetrics
Ilayer_regularization_losses
Jlayer_metrics
 	variables
!trainable_variables
"regularization_losses
$__call__
*%&call_and_return_all_conditional_losses
&%"call_and_return_conditional_losses*
6
Ktrace_0
Ltrace_1
Mtrace_2
Ntrace_3* 
6
Otrace_0
Ptrace_1
Qtrace_2
Rtrace_3* 
R
'0
S1
T2
U3
V4
W5
X6
Y7
Z8
[9
\10*
SM
VARIABLE_VALUE	iteration0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUElearning_rate3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
'
S0
U1
W2
Y3
[4*
'
T0
V1
X2
Z3
\4*
* 
* 
8
]	variables
^	keras_api
	_total
	`count*
8
a	variables
b	keras_api
	ctotal
	dcount*
H
e	variables
f	keras_api
	gtotal
	hcount
i
_fn_kwargs*

0
1
2*

0
1
2*
* 
Ъ

jstates
knon_trainable_variables

llayers
mmetrics
nlayer_regularization_losses
olayer_metrics
1	variables
2trainable_variables
3regularization_losses
5__call__
*6&call_and_return_all_conditional_losses
&6"call_and_return_conditional_losses*
6
ptrace_0
qtrace_1
rtrace_2
strace_3* 
6
ttrace_0
utrace_1
vtrace_2
wtrace_3* 
* 
с
x	variables
ytrainable_variables
zregularization_losses
{	keras_api
|__call__
*}&call_and_return_all_conditional_losses
~_random_generator

state_size

kernel
recurrent_kernel
bias*
* 

0
1*

0
1*
* 
ў
ђnon_trainable_variables
Ђlayers
ѓmetrics
 Ѓlayer_regularization_losses
ёlayer_metrics
:	variables
;trainable_variables
<regularization_losses
>__call__
*?&call_and_return_all_conditional_losses
&?"call_and_return_conditional_losses*

Ёtrace_0* 

єtrace_0* 
* 
* 
* 
ќ
Єnon_trainable_variables
ѕlayers
Ѕmetrics
 іlayer_regularization_losses
Іlayer_metrics
@	variables
Atrainable_variables
Bregularization_losses
D__call__
*E&call_and_return_all_conditional_losses
&E"call_and_return_conditional_losses* 

їtrace_0* 

Їtrace_0* 
* 

0
1
2*
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
mg
VARIABLE_VALUE"Adam/m/lstm_13/lstm_cell_13/kernel1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE"Adam/v/lstm_13/lstm_cell_13/kernel1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE,Adam/m/lstm_13/lstm_cell_13/recurrent_kernel1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE,Adam/v/lstm_13/lstm_cell_13/recurrent_kernel1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUE Adam/m/lstm_13/lstm_cell_13/bias1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUE Adam/v/lstm_13/lstm_cell_13/bias1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/m/dense_13/kernel1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/dense_13/kernel1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/m/dense_13/bias1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/dense_13/bias2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUE*

_0
`1*

]	variables*
UO
VARIABLE_VALUEtotal_24keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_24keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

c0
d1*

a	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*

g0
h1*

e	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 

80*
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

0
1
2*

0
1
2*
* 
ў
јnon_trainable_variables
Јlayers
љmetrics
 Љlayer_regularization_losses
њlayer_metrics
x	variables
ytrainable_variables
zregularization_losses
|__call__
*}&call_and_return_all_conditional_losses
&}"call_and_return_conditional_losses*
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
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
ј

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename/lstm_13/lstm_cell_13/kernel/Read/ReadVariableOp9lstm_13/lstm_cell_13/recurrent_kernel/Read/ReadVariableOp-lstm_13/lstm_cell_13/bias/Read/ReadVariableOp#dense_13/kernel/Read/ReadVariableOp!dense_13/bias/Read/ReadVariableOpiteration/Read/ReadVariableOp!learning_rate/Read/ReadVariableOp6Adam/m/lstm_13/lstm_cell_13/kernel/Read/ReadVariableOp6Adam/v/lstm_13/lstm_cell_13/kernel/Read/ReadVariableOp@Adam/m/lstm_13/lstm_cell_13/recurrent_kernel/Read/ReadVariableOp@Adam/v/lstm_13/lstm_cell_13/recurrent_kernel/Read/ReadVariableOp4Adam/m/lstm_13/lstm_cell_13/bias/Read/ReadVariableOp4Adam/v/lstm_13/lstm_cell_13/bias/Read/ReadVariableOp*Adam/m/dense_13/kernel/Read/ReadVariableOp*Adam/v/dense_13/kernel/Read/ReadVariableOp(Adam/m/dense_13/bias/Read/ReadVariableOp(Adam/v/dense_13/bias/Read/ReadVariableOptotal_2/Read/ReadVariableOpcount_2/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOpConst*$
Tin
2	*
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
GPU 2J 8ѓ *)
f$R"
 __inference__traced_save_1213993
й
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamelstm_13/lstm_cell_13/kernel%lstm_13/lstm_cell_13/recurrent_kernellstm_13/lstm_cell_13/biasdense_13/kerneldense_13/bias	iterationlearning_rate"Adam/m/lstm_13/lstm_cell_13/kernel"Adam/v/lstm_13/lstm_cell_13/kernel,Adam/m/lstm_13/lstm_cell_13/recurrent_kernel,Adam/v/lstm_13/lstm_cell_13/recurrent_kernel Adam/m/lstm_13/lstm_cell_13/bias Adam/v/lstm_13/lstm_cell_13/biasAdam/m/dense_13/kernelAdam/v/dense_13/kernelAdam/m/dense_13/biasAdam/v/dense_13/biastotal_2count_2total_1count_1totalcount*#
Tin
2*
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
GPU 2J 8ѓ *,
f'R%
#__inference__traced_restore_1214072ЁЗ4
┴B
═
*__inference_gpu_lstm_with_fallback_1210979

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4ѕc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:         P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : p

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0*,
_output_shapes
:         аR
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : t
ExpandDims_1
ExpandDimsinit_cExpandDims_1/dim:output:0*
T0*,
_output_shapes
:         аQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ї
splitSplitsplit/split_dim:output:0kernel*
T0*@
_output_shapes.
,:	а:	а:	а:	а*
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ъ
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*D
_output_shapes2
0:
аа:
аа:
аа:
аа*
	num_splite
zeros_like/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB:ђ	U
zeros_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    x

zeros_likeFill#zeros_like/shape_as_tensor:output:0zeros_like/Const:output:0*
T0*
_output_shapes	
:ђ	M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : r
concatConcatV2zeros_like:output:0biasconcat/axis:output:0*
N*
T0*
_output_shapes	
:ђS
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : Ц
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*L
_output_shapes:
8:а:а:а:а:а:а:а:а*
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
         a
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       m
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes
:	аY
ReshapeReshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:аa
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       m
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes
:	а[
	Reshape_1Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:аa
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       m
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes
:	а[
	Reshape_2Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:аa
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       m
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0*
_output_shapes
:	а[
	Reshape_3Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes	
:аa
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       p
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0* 
_output_shapes
:
аа\
	Reshape_4Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes

:ђѕa
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       p
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0* 
_output_shapes
:
аа\
	Reshape_5Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes

:ђѕa
transpose_7/permConst*
_output_shapes
:*
dtype0*
valueB"       p
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0* 
_output_shapes
:
аа\
	Reshape_6Reshapetranspose_7:y:0Const:output:0*
T0*
_output_shapes

:ђѕa
transpose_8/permConst*
_output_shapes
:*
dtype0*
valueB"       p
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0* 
_output_shapes
:
аа\
	Reshape_7Reshapetranspose_8:y:0Const:output:0*
T0*
_output_shapes

:ђѕ\
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes	
:а\
	Reshape_9Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes	
:а]

Reshape_10Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes	
:а]

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes	
:а]

Reshape_12Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes	
:а]

Reshape_13Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes	
:а]

Reshape_14Reshapesplit_2:output:6Const:output:0*
T0*
_output_shapes	
:а]

Reshape_15Reshapesplit_2:output:7Const:output:0*
T0*
_output_shapes	
:аO
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : а
concat_1ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T0*
_output_shapes

:ђ╗═
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1:output:0*
T0*`
_output_shapesN
L:         а:         а:         а:f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
         _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Т
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         а*
shrink_axis_maskq
SqueezeSqueezeCudnnRNN:output_h:0*
T0*(
_output_shapes
:         а*
squeeze_dims
 s
	Squeeze_1SqueezeCudnnRNN:output_c:0*
T0*(
_output_shapes
:         а*
squeeze_dims
 R
ExpandDims_2/dimConst*
_output_shapes
: *
dtype0*
value	B :ё
ExpandDims_2
ExpandDimsstrided_slice:output:0ExpandDims_2/dim:output:0*
T0*,
_output_shapes
:         а[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @_
IdentityIdentitystrided_slice:output:0*
T0*(
_output_shapes
:         аd

Identity_1IdentityExpandDims_2:output:0*
T0*,
_output_shapes
:         а[

Identity_2IdentitySqueeze:output:0*
T0*(
_output_shapes
:         а]

Identity_3IdentitySqueeze_1:output:0*
T0*(
_output_shapes
:         аI

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*p
_input_shapes_
]:         :         а:         а:	ђ	:
ађ	:ђ	*=
api_implements+)lstm_55c05eb7-11a9-4616-8925-c9051a66537d*
api_preferred_deviceGPU*
go_backwards( *

time_major( :S O
+
_output_shapes
:         
 
_user_specified_nameinputs:PL
(
_output_shapes
:         а
 
_user_specified_nameinit_h:PL
(
_output_shapes
:         а
 
_user_specified_nameinit_c:GC

_output_shapes
:	ђ	
 
_user_specified_namekernel:RN
 
_output_shapes
:
ађ	
*
_user_specified_namerecurrent_kernel:A=

_output_shapes	
:ђ	

_user_specified_namebias
џ	
к
while_cond_1210796
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice5
1while_while_cond_1210796___redundant_placeholder05
1while_while_cond_1210796___redundant_placeholder15
1while_while_cond_1210796___redundant_placeholder25
1while_while_cond_1210796___redundant_placeholder3
while_identity
`

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :         а:         а: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:         а:.*
(
_output_shapes
:         а:

_output_shapes
: :

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
Јd
▄
#__inference__traced_restore_1214072
file_prefix?
,assignvariableop_lstm_13_lstm_cell_13_kernel:	ђ	L
8assignvariableop_1_lstm_13_lstm_cell_13_recurrent_kernel:
ађ	;
,assignvariableop_2_lstm_13_lstm_cell_13_bias:	ђ	5
"assignvariableop_3_dense_13_kernel:	а.
 assignvariableop_4_dense_13_bias:&
assignvariableop_5_iteration:	 *
 assignvariableop_6_learning_rate: H
5assignvariableop_7_adam_m_lstm_13_lstm_cell_13_kernel:	ђ	H
5assignvariableop_8_adam_v_lstm_13_lstm_cell_13_kernel:	ђ	S
?assignvariableop_9_adam_m_lstm_13_lstm_cell_13_recurrent_kernel:
ађ	T
@assignvariableop_10_adam_v_lstm_13_lstm_cell_13_recurrent_kernel:
ађ	C
4assignvariableop_11_adam_m_lstm_13_lstm_cell_13_bias:	ђ	C
4assignvariableop_12_adam_v_lstm_13_lstm_cell_13_bias:	ђ	=
*assignvariableop_13_adam_m_dense_13_kernel:	а=
*assignvariableop_14_adam_v_dense_13_kernel:	а6
(assignvariableop_15_adam_m_dense_13_bias:6
(assignvariableop_16_adam_v_dense_13_bias:%
assignvariableop_17_total_2: %
assignvariableop_18_count_2: %
assignvariableop_19_total_1: %
assignvariableop_20_count_1: #
assignvariableop_21_total: #
assignvariableop_22_count: 
identity_24ѕбAssignVariableOpбAssignVariableOp_1бAssignVariableOp_10бAssignVariableOp_11бAssignVariableOp_12бAssignVariableOp_13бAssignVariableOp_14бAssignVariableOp_15бAssignVariableOp_16бAssignVariableOp_17бAssignVariableOp_18бAssignVariableOp_19бAssignVariableOp_2бAssignVariableOp_20бAssignVariableOp_21бAssignVariableOp_22бAssignVariableOp_3бAssignVariableOp_4бAssignVariableOp_5бAssignVariableOp_6бAssignVariableOp_7бAssignVariableOp_8бAssignVariableOp_9 	
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Ц	
valueЏ	Bў	B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHа
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*C
value:B8B B B B B B B B B B B B B B B B B B B B B B B B ќ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*t
_output_shapesb
`::::::::::::::::::::::::*&
dtypes
2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:┐
AssignVariableOpAssignVariableOp,assignvariableop_lstm_13_lstm_cell_13_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:¤
AssignVariableOp_1AssignVariableOp8assignvariableop_1_lstm_13_lstm_cell_13_recurrent_kernelIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:├
AssignVariableOp_2AssignVariableOp,assignvariableop_2_lstm_13_lstm_cell_13_biasIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:╣
AssignVariableOp_3AssignVariableOp"assignvariableop_3_dense_13_kernelIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:и
AssignVariableOp_4AssignVariableOp assignvariableop_4_dense_13_biasIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0	*
_output_shapes
:│
AssignVariableOp_5AssignVariableOpassignvariableop_5_iterationIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:и
AssignVariableOp_6AssignVariableOp assignvariableop_6_learning_rateIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:╠
AssignVariableOp_7AssignVariableOp5assignvariableop_7_adam_m_lstm_13_lstm_cell_13_kernelIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:╠
AssignVariableOp_8AssignVariableOp5assignvariableop_8_adam_v_lstm_13_lstm_cell_13_kernelIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:о
AssignVariableOp_9AssignVariableOp?assignvariableop_9_adam_m_lstm_13_lstm_cell_13_recurrent_kernelIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:┘
AssignVariableOp_10AssignVariableOp@assignvariableop_10_adam_v_lstm_13_lstm_cell_13_recurrent_kernelIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:═
AssignVariableOp_11AssignVariableOp4assignvariableop_11_adam_m_lstm_13_lstm_cell_13_biasIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:═
AssignVariableOp_12AssignVariableOp4assignvariableop_12_adam_v_lstm_13_lstm_cell_13_biasIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:├
AssignVariableOp_13AssignVariableOp*assignvariableop_13_adam_m_dense_13_kernelIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:├
AssignVariableOp_14AssignVariableOp*assignvariableop_14_adam_v_dense_13_kernelIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:┴
AssignVariableOp_15AssignVariableOp(assignvariableop_15_adam_m_dense_13_biasIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:┴
AssignVariableOp_16AssignVariableOp(assignvariableop_16_adam_v_dense_13_biasIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:┤
AssignVariableOp_17AssignVariableOpassignvariableop_17_total_2Identity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:┤
AssignVariableOp_18AssignVariableOpassignvariableop_18_count_2Identity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:┤
AssignVariableOp_19AssignVariableOpassignvariableop_19_total_1Identity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:┤
AssignVariableOp_20AssignVariableOpassignvariableop_20_count_1Identity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:▓
AssignVariableOp_21AssignVariableOpassignvariableop_21_totalIdentity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:▓
AssignVariableOp_22AssignVariableOpassignvariableop_22_countIdentity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 ╔
Identity_23Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_24IdentityIdentity_23:output:0^NoOp_1*
T0*
_output_shapes
: Х
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_24Identity_24:output:0*C
_input_shapes2
0: : : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_22AssignVariableOp_222(
AssignVariableOp_3AssignVariableOp_32(
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
эM
б
(__forward_gpu_lstm_with_fallback_1209484

inputs
init_h_0
init_c_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4
strided_slice
cudnnrnn

cudnnrnn_0

cudnnrnn_1

cudnnrnn_2
	transpose

expanddims
expanddims_1
concat_1
transpose_perm

init_h

init_c
concat_1_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
transpose_7_perm
transpose_8_perm
split_2_split_dim
split_split_dim
split_1_split_dim
concat_axisѕc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          B
transpose_0	Transposeinputstranspose/perm:output:0*
T0P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : r

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T0*,
_output_shapes
:         аR
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : v
ExpandDims_1
ExpandDimsinit_c_0ExpandDims_1/dim:output:0*
T0*,
_output_shapes
:         аQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ї
splitSplitsplit/split_dim:output:0kernel*
T0*@
_output_shapes.
,:	а:	а:	а:	а*
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ъ
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*D
_output_shapes2
0:
аа:
аа:
аа:
аа*
	num_splite
zeros_like/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB:ђ	U
zeros_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    x

zeros_likeFill#zeros_like/shape_as_tensor:output:0zeros_like/Const:output:0*
T0*
_output_shapes	
:ђ	M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : r
concatConcatV2zeros_like:output:0biasconcat/axis:output:0*
N*
T0*
_output_shapes	
:ђS
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : Ц
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*L
_output_shapes:
8:а:а:а:а:а:а:а:а*
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
         a
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       m
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes
:	аY
ReshapeReshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:аa
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       m
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes
:	а[
	Reshape_1Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:аa
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       m
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes
:	а[
	Reshape_2Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:аa
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       m
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0*
_output_shapes
:	а[
	Reshape_3Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes	
:аa
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       p
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0* 
_output_shapes
:
аа\
	Reshape_4Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes

:ђѕa
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       p
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0* 
_output_shapes
:
аа\
	Reshape_5Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes

:ђѕa
transpose_7/permConst*
_output_shapes
:*
dtype0*
valueB"       p
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0* 
_output_shapes
:
аа\
	Reshape_6Reshapetranspose_7:y:0Const:output:0*
T0*
_output_shapes

:ђѕa
transpose_8/permConst*
_output_shapes
:*
dtype0*
valueB"       p
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0* 
_output_shapes
:
аа\
	Reshape_7Reshapetranspose_8:y:0Const:output:0*
T0*
_output_shapes

:ђѕ\
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes	
:а\
	Reshape_9Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes	
:а]

Reshape_10Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes	
:а]

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes	
:а]

Reshape_12Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes	
:а]

Reshape_13Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes	
:а]

Reshape_14Reshapesplit_2:output:6Const:output:0*
T0*
_output_shapes	
:а]

Reshape_15Reshapesplit_2:output:7Const:output:0*
T0*
_output_shapes	
:аO
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ё

concat_1_0ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T0Л
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1_0:output:0*
T0*`
_output_shapesN
L:         а:         а:         а:f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
         _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Й
strided_slice_0StridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
shrink_axis_maskq
SqueezeSqueezeCudnnRNN:output_h:0*
T0*(
_output_shapes
:         а*
squeeze_dims
 s
	Squeeze_1SqueezeCudnnRNN:output_c:0*
T0*(
_output_shapes
:         а*
squeeze_dims
 R
ExpandDims_2/dimConst*
_output_shapes
: *
dtype0*
value	B :є
ExpandDims_2
ExpandDimsstrided_slice_0:output:0ExpandDims_2/dim:output:0*
T0*,
_output_shapes
:         а[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @a
IdentityIdentitystrided_slice_0:output:0*
T0*(
_output_shapes
:         аd

Identity_1IdentityExpandDims_2:output:0*
T0*,
_output_shapes
:         а[

Identity_2IdentitySqueeze:output:0*
T0*(
_output_shapes
:         а]

Identity_3IdentitySqueeze_1:output:0*
T0*(
_output_shapes
:         аI

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: "
concat_1concat_1_0:output:0"'
concat_1_axisconcat_1/axis:output:0"#
concat_axisconcat/axis:output:0"
cudnnrnnCudnnRNN:output_h:0"!

cudnnrnn_0CudnnRNN:output_c:0"

cudnnrnn_1CudnnRNN:output:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"!

expanddimsExpandDims:output:0"%
expanddims_1ExpandDims_1:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
init_cinit_c_0"
init_hinit_h_0"/
split_1_split_dimsplit_1/split_dim:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"+
split_split_dimsplit/split_dim:output:0")
strided_slicestrided_slice_0:output:0"
	transposetranspose_0:y:0"-
transpose_1_permtranspose_1/perm:output:0"-
transpose_2_permtranspose_2/perm:output:0"-
transpose_3_permtranspose_3/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"-
transpose_5_permtranspose_5/perm:output:0"-
transpose_6_permtranspose_6/perm:output:0"-
transpose_7_permtranspose_7/perm:output:0"-
transpose_8_permtranspose_8/perm:output:0")
transpose_permtranspose/perm:output:0*(
_construction_contextkEagerRuntime*p
_input_shapes_
]:         :         а:         а:	ђ	:
ађ	:ђ	*=
api_implements+)lstm_4726b1d6-6e07-45d8-b9f1-158de8b4230f*
api_preferred_deviceGPU*Y
backward_function_name?=__inference___backward_gpu_lstm_with_fallback_1209309_1209485*
go_backwards( *

time_major( :S O
+
_output_shapes
:         
 
_user_specified_nameinputs:PL
(
_output_shapes
:         а
 
_user_specified_nameinit_h:PL
(
_output_shapes
:         а
 
_user_specified_nameinit_c:GC

_output_shapes
:	ђ	
 
_user_specified_namekernel:RN
 
_output_shapes
:
ађ	
*
_user_specified_namerecurrent_kernel:A=

_output_shapes	
:ђ	

_user_specified_namebias
░5
┼

 __inference__traced_save_1213993
file_prefix:
6savev2_lstm_13_lstm_cell_13_kernel_read_readvariableopD
@savev2_lstm_13_lstm_cell_13_recurrent_kernel_read_readvariableop8
4savev2_lstm_13_lstm_cell_13_bias_read_readvariableop.
*savev2_dense_13_kernel_read_readvariableop,
(savev2_dense_13_bias_read_readvariableop(
$savev2_iteration_read_readvariableop	,
(savev2_learning_rate_read_readvariableopA
=savev2_adam_m_lstm_13_lstm_cell_13_kernel_read_readvariableopA
=savev2_adam_v_lstm_13_lstm_cell_13_kernel_read_readvariableopK
Gsavev2_adam_m_lstm_13_lstm_cell_13_recurrent_kernel_read_readvariableopK
Gsavev2_adam_v_lstm_13_lstm_cell_13_recurrent_kernel_read_readvariableop?
;savev2_adam_m_lstm_13_lstm_cell_13_bias_read_readvariableop?
;savev2_adam_v_lstm_13_lstm_cell_13_bias_read_readvariableop5
1savev2_adam_m_dense_13_kernel_read_readvariableop5
1savev2_adam_v_dense_13_kernel_read_readvariableop3
/savev2_adam_m_dense_13_bias_read_readvariableop3
/savev2_adam_v_dense_13_bias_read_readvariableop&
"savev2_total_2_read_readvariableop&
"savev2_count_2_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop
savev2_const

identity_1ѕбMergeV2Checkpointsw
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
_temp/partЂ
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
value	B : Њ
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: Ч	
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Ц	
valueЏ	Bў	B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHЮ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*C
value:B8B B B B B B B B B B B B B B B B B B B B B B B B в

SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:06savev2_lstm_13_lstm_cell_13_kernel_read_readvariableop@savev2_lstm_13_lstm_cell_13_recurrent_kernel_read_readvariableop4savev2_lstm_13_lstm_cell_13_bias_read_readvariableop*savev2_dense_13_kernel_read_readvariableop(savev2_dense_13_bias_read_readvariableop$savev2_iteration_read_readvariableop(savev2_learning_rate_read_readvariableop=savev2_adam_m_lstm_13_lstm_cell_13_kernel_read_readvariableop=savev2_adam_v_lstm_13_lstm_cell_13_kernel_read_readvariableopGsavev2_adam_m_lstm_13_lstm_cell_13_recurrent_kernel_read_readvariableopGsavev2_adam_v_lstm_13_lstm_cell_13_recurrent_kernel_read_readvariableop;savev2_adam_m_lstm_13_lstm_cell_13_bias_read_readvariableop;savev2_adam_v_lstm_13_lstm_cell_13_bias_read_readvariableop1savev2_adam_m_dense_13_kernel_read_readvariableop1savev2_adam_v_dense_13_kernel_read_readvariableop/savev2_adam_m_dense_13_bias_read_readvariableop/savev2_adam_v_dense_13_bias_read_readvariableop"savev2_total_2_read_readvariableop"savev2_count_2_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableopsavev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *&
dtypes
2	љ
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:│
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
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

identity_1Identity_1:output:0*Х
_input_shapesц
А: :	ђ	:
ађ	:ђ	:	а:: : :	ђ	:	ђ	:
ађ	:
ађ	:ђ	:ђ	:	а:	а::: : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	ђ	:&"
 
_output_shapes
:
ађ	:!

_output_shapes	
:ђ	:%!

_output_shapes
:	а: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	ђ	:%	!

_output_shapes
:	ђ	:&
"
 
_output_shapes
:
ађ	:&"
 
_output_shapes
:
ађ	:!

_output_shapes	
:ђ	:!

_output_shapes	
:ђ	:%!

_output_shapes
:	а:%!

_output_shapes
:	а: 

_output_shapes
:: 

_output_shapes
::
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
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
┴B
═
*__inference_gpu_lstm_with_fallback_1207969

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4ѕc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:         P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : p

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0*,
_output_shapes
:         аR
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : t
ExpandDims_1
ExpandDimsinit_cExpandDims_1/dim:output:0*
T0*,
_output_shapes
:         аQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ї
splitSplitsplit/split_dim:output:0kernel*
T0*@
_output_shapes.
,:	а:	а:	а:	а*
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ъ
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*D
_output_shapes2
0:
аа:
аа:
аа:
аа*
	num_splite
zeros_like/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB:ђ	U
zeros_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    x

zeros_likeFill#zeros_like/shape_as_tensor:output:0zeros_like/Const:output:0*
T0*
_output_shapes	
:ђ	M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : r
concatConcatV2zeros_like:output:0biasconcat/axis:output:0*
N*
T0*
_output_shapes	
:ђS
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : Ц
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*L
_output_shapes:
8:а:а:а:а:а:а:а:а*
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
         a
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       m
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes
:	аY
ReshapeReshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:аa
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       m
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes
:	а[
	Reshape_1Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:аa
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       m
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes
:	а[
	Reshape_2Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:аa
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       m
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0*
_output_shapes
:	а[
	Reshape_3Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes	
:аa
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       p
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0* 
_output_shapes
:
аа\
	Reshape_4Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes

:ђѕa
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       p
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0* 
_output_shapes
:
аа\
	Reshape_5Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes

:ђѕa
transpose_7/permConst*
_output_shapes
:*
dtype0*
valueB"       p
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0* 
_output_shapes
:
аа\
	Reshape_6Reshapetranspose_7:y:0Const:output:0*
T0*
_output_shapes

:ђѕa
transpose_8/permConst*
_output_shapes
:*
dtype0*
valueB"       p
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0* 
_output_shapes
:
аа\
	Reshape_7Reshapetranspose_8:y:0Const:output:0*
T0*
_output_shapes

:ђѕ\
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes	
:а\
	Reshape_9Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes	
:а]

Reshape_10Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes	
:а]

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes	
:а]

Reshape_12Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes	
:а]

Reshape_13Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes	
:а]

Reshape_14Reshapesplit_2:output:6Const:output:0*
T0*
_output_shapes	
:а]

Reshape_15Reshapesplit_2:output:7Const:output:0*
T0*
_output_shapes	
:аO
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : а
concat_1ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T0*
_output_shapes

:ђ╗═
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1:output:0*
T0*`
_output_shapesN
L:         а:         а:         а:f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
         _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Т
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         а*
shrink_axis_maskq
SqueezeSqueezeCudnnRNN:output_h:0*
T0*(
_output_shapes
:         а*
squeeze_dims
 s
	Squeeze_1SqueezeCudnnRNN:output_c:0*
T0*(
_output_shapes
:         а*
squeeze_dims
 R
ExpandDims_2/dimConst*
_output_shapes
: *
dtype0*
value	B :ё
ExpandDims_2
ExpandDimsstrided_slice:output:0ExpandDims_2/dim:output:0*
T0*,
_output_shapes
:         а[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @_
IdentityIdentitystrided_slice:output:0*
T0*(
_output_shapes
:         аd

Identity_1IdentityExpandDims_2:output:0*
T0*,
_output_shapes
:         а[

Identity_2IdentitySqueeze:output:0*
T0*(
_output_shapes
:         а]

Identity_3IdentitySqueeze_1:output:0*
T0*(
_output_shapes
:         аI

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*p
_input_shapes_
]:         :         а:         а:	ђ	:
ађ	:ђ	*=
api_implements+)lstm_e15ea421-5ae9-42eb-bcec-970c1aedc419*
api_preferred_deviceGPU*
go_backwards( *

time_major( :S O
+
_output_shapes
:         
 
_user_specified_nameinputs:PL
(
_output_shapes
:         а
 
_user_specified_nameinit_h:PL
(
_output_shapes
:         а
 
_user_specified_nameinit_c:GC

_output_shapes
:	ђ	
 
_user_specified_namekernel:RN
 
_output_shapes
:
ађ	
*
_user_specified_namerecurrent_kernel:A=

_output_shapes	
:ђ	

_user_specified_namebias
ф┬
у
=__inference___backward_gpu_lstm_with_fallback_1211457_1211633
placeholder
placeholder_1
placeholder_2
placeholder_3
placeholder_43
/gradients_expanddims_2_grad_shape_strided_slice)
%gradients_squeeze_grad_shape_cudnnrnn+
'gradients_squeeze_1_grad_shape_cudnnrnn/
+gradients_strided_slice_grad_shape_cudnnrnn!
gradients_zeros_like_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9
5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_15
1gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h,
(gradients_expanddims_1_grad_shape_init_c-
)gradients_concat_1_grad_mod_concat_1_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_permA
=gradients_transpose_7_grad_invertpermutation_transpose_7_permA
=gradients_transpose_8_grad_invertpermutation_transpose_8_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim)
%gradients_concat_grad_mod_concat_axis
identity

identity_1

identity_2

identity_3

identity_4

identity_5ѕ_
gradients/grad_ys_0Identityplaceholder*
T0*(
_output_shapes
:         аe
gradients/grad_ys_1Identityplaceholder_1*
T0*,
_output_shapes
:         аa
gradients/grad_ys_2Identityplaceholder_2*
T0*(
_output_shapes
:         аa
gradients/grad_ys_3Identityplaceholder_3*
T0*(
_output_shapes
:         аO
gradients/grad_ys_4Identityplaceholder_4*
T0*
_output_shapes
: ђ
!gradients/ExpandDims_2_grad/ShapeShape/gradients_expanddims_2_grad_shape_strided_slice*
T0*
_output_shapes
:Ф
#gradients/ExpandDims_2_grad/ReshapeReshapegradients/grad_ys_1:output:0*gradients/ExpandDims_2_grad/Shape:output:0*
T0*(
_output_shapes
:         аq
gradients/Squeeze_grad/ShapeShape%gradients_squeeze_grad_shape_cudnnrnn*
T0*
_output_shapes
:Ц
gradients/Squeeze_grad/ReshapeReshapegradients/grad_ys_2:output:0%gradients/Squeeze_grad/Shape:output:0*
T0*,
_output_shapes
:         аu
gradients/Squeeze_1_grad/ShapeShape'gradients_squeeze_1_grad_shape_cudnnrnn*
T0*
_output_shapes
:Е
 gradients/Squeeze_1_grad/ReshapeReshapegradients/grad_ys_3:output:0'gradients/Squeeze_1_grad/Shape:output:0*
T0*,
_output_shapes
:         ак
gradients/AddNAddNgradients/grad_ys_0:output:0,gradients/ExpandDims_2_grad/Reshape:output:0*
N*
T0*&
_class
loc:@gradients/grad_ys_0*(
_output_shapes
:         а}
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
:є
3gradients/strided_slice_grad/StridedSliceGrad/beginConst*
_output_shapes
:*
dtype0*
valueB:
         {
1gradients/strided_slice_grad/StridedSliceGrad/endConst*
_output_shapes
:*
dtype0*
valueB: 
5gradients/strided_slice_grad/StridedSliceGrad/stridesConst*
_output_shapes
:*
dtype0*
valueB:џ
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0<gradients/strided_slice_grad/StridedSliceGrad/begin:output:0:gradients/strided_slice_grad/StridedSliceGrad/end:output:0>gradients/strided_slice_grad/StridedSliceGrad/strides:output:0gradients/AddN:sum:0*
Index0*
T0*,
_output_shapes
:         а*
shrink_axis_maskc
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
:Ц
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_11gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1+gradients_strided_slice_grad_shape_cudnnrnn%gradients_squeeze_grad_shape_cudnnrnn'gradients_squeeze_1_grad_shape_cudnnrnn6gradients/strided_slice_grad/StridedSliceGrad:output:0'gradients/Squeeze_grad/Reshape:output:0)gradients/Squeeze_1_grad/Reshape:output:0gradients_zeros_like_cudnnrnn*
T0*c
_output_shapesQ
O:         :         а:         а:ђ╗ќ
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:л
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*+
_output_shapes
:         u
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
:к
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*(
_output_shapes
:         аy
!gradients/ExpandDims_1_grad/ShapeShape(gradients_expanddims_1_grad_shape_init_c*
T0*
_output_shapes
:╩
#gradients/ExpandDims_1_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_c_backprop:0*gradients/ExpandDims_1_grad/Shape:output:0*
T0*(
_output_shapes
:         а^
gradients/concat_1_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :џ
gradients/concat_1_grad/modFloorMod)gradients_concat_1_grad_mod_concat_1_axis%gradients/concat_1_grad/Rank:output:0*
T0*
_output_shapes
: h
gradients/concat_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:аj
gradients/concat_1_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:аj
gradients/concat_1_grad/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:аj
gradients/concat_1_grad/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:аk
gradients/concat_1_grad/Shape_4Const*
_output_shapes
:*
dtype0*
valueB:ђѕk
gradients/concat_1_grad/Shape_5Const*
_output_shapes
:*
dtype0*
valueB:ђѕk
gradients/concat_1_grad/Shape_6Const*
_output_shapes
:*
dtype0*
valueB:ђѕk
gradients/concat_1_grad/Shape_7Const*
_output_shapes
:*
dtype0*
valueB:ђѕj
gradients/concat_1_grad/Shape_8Const*
_output_shapes
:*
dtype0*
valueB:аj
gradients/concat_1_grad/Shape_9Const*
_output_shapes
:*
dtype0*
valueB:аk
 gradients/concat_1_grad/Shape_10Const*
_output_shapes
:*
dtype0*
valueB:аk
 gradients/concat_1_grad/Shape_11Const*
_output_shapes
:*
dtype0*
valueB:аk
 gradients/concat_1_grad/Shape_12Const*
_output_shapes
:*
dtype0*
valueB:аk
 gradients/concat_1_grad/Shape_13Const*
_output_shapes
:*
dtype0*
valueB:аk
 gradients/concat_1_grad/Shape_14Const*
_output_shapes
:*
dtype0*
valueB:аk
 gradients/concat_1_grad/Shape_15Const*
_output_shapes
:*
dtype0*
valueB:аЭ
$gradients/concat_1_grad/ConcatOffsetConcatOffsetgradients/concat_1_grad/mod:z:0&gradients/concat_1_grad/Shape:output:0(gradients/concat_1_grad/Shape_1:output:0(gradients/concat_1_grad/Shape_2:output:0(gradients/concat_1_grad/Shape_3:output:0(gradients/concat_1_grad/Shape_4:output:0(gradients/concat_1_grad/Shape_5:output:0(gradients/concat_1_grad/Shape_6:output:0(gradients/concat_1_grad/Shape_7:output:0(gradients/concat_1_grad/Shape_8:output:0(gradients/concat_1_grad/Shape_9:output:0)gradients/concat_1_grad/Shape_10:output:0)gradients/concat_1_grad/Shape_11:output:0)gradients/concat_1_grad/Shape_12:output:0)gradients/concat_1_grad/Shape_13:output:0)gradients/concat_1_grad/Shape_14:output:0)gradients/concat_1_grad/Shape_15:output:0*
N*t
_output_shapesb
`::::::::::::::::В
gradients/concat_1_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:0&gradients/concat_1_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:а­
gradients/concat_1_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:1(gradients/concat_1_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:а­
gradients/concat_1_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:2(gradients/concat_1_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes	
:а­
gradients/concat_1_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:3(gradients/concat_1_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes	
:аы
gradients/concat_1_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:4(gradients/concat_1_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes

:ђѕы
gradients/concat_1_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:5(gradients/concat_1_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes

:ђѕы
gradients/concat_1_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:6(gradients/concat_1_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes

:ђѕы
gradients/concat_1_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:7(gradients/concat_1_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes

:ђѕ­
gradients/concat_1_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:8(gradients/concat_1_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes	
:а­
gradients/concat_1_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:9(gradients/concat_1_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes	
:аз
 gradients/concat_1_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:10)gradients/concat_1_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes	
:аз
 gradients/concat_1_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:11)gradients/concat_1_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes	
:аз
 gradients/concat_1_grad/Slice_12Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:12)gradients/concat_1_grad/Shape_12:output:0*
Index0*
T0*
_output_shapes	
:аз
 gradients/concat_1_grad/Slice_13Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:13)gradients/concat_1_grad/Shape_13:output:0*
Index0*
T0*
_output_shapes	
:аз
 gradients/concat_1_grad/Slice_14Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:14)gradients/concat_1_grad/Shape_14:output:0*
Index0*
T0*
_output_shapes	
:аз
 gradients/concat_1_grad/Slice_15Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:15)gradients/concat_1_grad/Shape_15:output:0*
Index0*
T0*
_output_shapes	
:аm
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      б
gradients/Reshape_grad/ReshapeReshape&gradients/concat_1_grad/Slice:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes
:	аo
gradients/Reshape_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      е
 gradients/Reshape_1_grad/ReshapeReshape(gradients/concat_1_grad/Slice_1:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0*
_output_shapes
:	аo
gradients/Reshape_2_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      е
 gradients/Reshape_2_grad/ReshapeReshape(gradients/concat_1_grad/Slice_2:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0*
_output_shapes
:	аo
gradients/Reshape_3_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      е
 gradients/Reshape_3_grad/ReshapeReshape(gradients/concat_1_grad/Slice_3:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0*
_output_shapes
:	аo
gradients/Reshape_4_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      Е
 gradients/Reshape_4_grad/ReshapeReshape(gradients/concat_1_grad/Slice_4:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0* 
_output_shapes
:
ааo
gradients/Reshape_5_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      Е
 gradients/Reshape_5_grad/ReshapeReshape(gradients/concat_1_grad/Slice_5:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0* 
_output_shapes
:
ааo
gradients/Reshape_6_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      Е
 gradients/Reshape_6_grad/ReshapeReshape(gradients/concat_1_grad/Slice_6:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0* 
_output_shapes
:
ааo
gradients/Reshape_7_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      Е
 gradients/Reshape_7_grad/ReshapeReshape(gradients/concat_1_grad/Slice_7:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0* 
_output_shapes
:
ааi
gradients/Reshape_8_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:ац
 gradients/Reshape_8_grad/ReshapeReshape(gradients/concat_1_grad/Slice_8:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes	
:аi
gradients/Reshape_9_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:ац
 gradients/Reshape_9_grad/ReshapeReshape(gradients/concat_1_grad/Slice_9:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes	
:аj
gradients/Reshape_10_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:аД
!gradients/Reshape_10_grad/ReshapeReshape)gradients/concat_1_grad/Slice_10:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes	
:аj
gradients/Reshape_11_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:аД
!gradients/Reshape_11_grad/ReshapeReshape)gradients/concat_1_grad/Slice_11:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes	
:аj
gradients/Reshape_12_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:аД
!gradients/Reshape_12_grad/ReshapeReshape)gradients/concat_1_grad/Slice_12:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes	
:аj
gradients/Reshape_13_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:аД
!gradients/Reshape_13_grad/ReshapeReshape)gradients/concat_1_grad/Slice_13:output:0(gradients/Reshape_13_grad/Shape:output:0*
T0*
_output_shapes	
:аj
gradients/Reshape_14_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:аД
!gradients/Reshape_14_grad/ReshapeReshape)gradients/concat_1_grad/Slice_14:output:0(gradients/Reshape_14_grad/Shape:output:0*
T0*
_output_shapes	
:аj
gradients/Reshape_15_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:аД
!gradients/Reshape_15_grad/ReshapeReshape)gradients/concat_1_grad/Slice_15:output:0(gradients/Reshape_15_grad/Shape:output:0*
T0*
_output_shapes	
:аю
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:Х
$gradients/transpose_1_grad/transpose	Transpose'gradients/Reshape_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0*
_output_shapes
:	аю
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:И
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0*
_output_shapes
:	аю
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:И
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0*
_output_shapes
:	аю
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:И
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0*
_output_shapes
:	аю
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:╣
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
ааю
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:╣
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
ааю
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:╣
$gradients/transpose_7_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
ааю
,gradients/transpose_8_grad/InvertPermutationInvertPermutation=gradients_transpose_8_grad_invertpermutation_transpose_8_perm*
_output_shapes
:╣
$gradients/transpose_8_grad/transpose	Transpose)gradients/Reshape_7_grad/Reshape:output:00gradients/transpose_8_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
аау
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0*gradients/Reshape_13_grad/Reshape:output:0*gradients/Reshape_14_grad/Reshape:output:0*gradients/Reshape_15_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
N*
T0*
_output_shapes	
:ђ»
gradients/split_grad/concatConcatV2(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
N*
T0*
_output_shapes
:	ђ	Х
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0(gradients/transpose_7_grad/transpose:y:0(gradients/transpose_8_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
N*
T0* 
_output_shapes
:
ађ	\
gradients/concat_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :њ
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: f
gradients/concat_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:ђ	h
gradients/concat_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:ђ	╩
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0*
N* 
_output_shapes
::м
gradients/concat_grad/SliceSlice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:ђ	о
gradients/concat_grad/Slice_1Slice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:ђ	r
IdentityIdentity&gradients/transpose_grad/transpose:y:0*
T0*+
_output_shapes
:         u

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0*
T0*(
_output_shapes
:         аw

Identity_2Identity,gradients/ExpandDims_1_grad/Reshape:output:0*
T0*(
_output_shapes
:         аf

Identity_3Identity$gradients/split_grad/concat:output:0*
T0*
_output_shapes
:	ђ	i

Identity_4Identity&gradients/split_1_grad/concat:output:0*
T0* 
_output_shapes
:
ађ	d

Identity_5Identity&gradients/concat_grad/Slice_1:output:0*
T0*
_output_shapes	
:ђ	"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*(
_construction_contextkEagerRuntime*ѓ
_input_shapes­
ь:         а:         а:         а:         а: :         а:         а:         а:         а::         :         а:         а:ђ╗::         а:         а: ::::::::: : : : *=
api_implements+)lstm_cbe182d0-608f-49e0-986e-6c0a38bbebc2*
api_preferred_deviceGPU*C
forward_function_name*(__forward_gpu_lstm_with_fallback_1211632*
go_backwards( *

time_major( :. *
(
_output_shapes
:         а:2.
,
_output_shapes
:         а:.*
(
_output_shapes
:         а:.*
(
_output_shapes
:         а:

_output_shapes
: :.*
(
_output_shapes
:         а:2.
,
_output_shapes
:         а:2.
,
_output_shapes
:         а:2.
,
_output_shapes
:         а:	

_output_shapes
::1
-
+
_output_shapes
:         :2.
,
_output_shapes
:         а:2.
,
_output_shapes
:         а:"

_output_shapes

:ђ╗: 

_output_shapes
::.*
(
_output_shapes
:         а:.*
(
_output_shapes
:         а:

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
эM
б
(__forward_gpu_lstm_with_fallback_1212078

inputs
init_h_0
init_c_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4
strided_slice
cudnnrnn

cudnnrnn_0

cudnnrnn_1

cudnnrnn_2
	transpose

expanddims
expanddims_1
concat_1
transpose_perm

init_h

init_c
concat_1_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
transpose_7_perm
transpose_8_perm
split_2_split_dim
split_split_dim
split_1_split_dim
concat_axisѕc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          B
transpose_0	Transposeinputstranspose/perm:output:0*
T0P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : r

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T0*,
_output_shapes
:         аR
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : v
ExpandDims_1
ExpandDimsinit_c_0ExpandDims_1/dim:output:0*
T0*,
_output_shapes
:         аQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ї
splitSplitsplit/split_dim:output:0kernel*
T0*@
_output_shapes.
,:	а:	а:	а:	а*
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ъ
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*D
_output_shapes2
0:
аа:
аа:
аа:
аа*
	num_splite
zeros_like/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB:ђ	U
zeros_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    x

zeros_likeFill#zeros_like/shape_as_tensor:output:0zeros_like/Const:output:0*
T0*
_output_shapes	
:ђ	M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : r
concatConcatV2zeros_like:output:0biasconcat/axis:output:0*
N*
T0*
_output_shapes	
:ђS
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : Ц
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*L
_output_shapes:
8:а:а:а:а:а:а:а:а*
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
         a
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       m
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes
:	аY
ReshapeReshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:аa
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       m
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes
:	а[
	Reshape_1Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:аa
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       m
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes
:	а[
	Reshape_2Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:аa
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       m
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0*
_output_shapes
:	а[
	Reshape_3Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes	
:аa
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       p
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0* 
_output_shapes
:
аа\
	Reshape_4Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes

:ђѕa
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       p
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0* 
_output_shapes
:
аа\
	Reshape_5Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes

:ђѕa
transpose_7/permConst*
_output_shapes
:*
dtype0*
valueB"       p
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0* 
_output_shapes
:
аа\
	Reshape_6Reshapetranspose_7:y:0Const:output:0*
T0*
_output_shapes

:ђѕa
transpose_8/permConst*
_output_shapes
:*
dtype0*
valueB"       p
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0* 
_output_shapes
:
аа\
	Reshape_7Reshapetranspose_8:y:0Const:output:0*
T0*
_output_shapes

:ђѕ\
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes	
:а\
	Reshape_9Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes	
:а]

Reshape_10Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes	
:а]

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes	
:а]

Reshape_12Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes	
:а]

Reshape_13Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes	
:а]

Reshape_14Reshapesplit_2:output:6Const:output:0*
T0*
_output_shapes	
:а]

Reshape_15Reshapesplit_2:output:7Const:output:0*
T0*
_output_shapes	
:аO
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ё

concat_1_0ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T0Л
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1_0:output:0*
T0*`
_output_shapesN
L:         а:         а:         а:f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
         _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Й
strided_slice_0StridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
shrink_axis_maskq
SqueezeSqueezeCudnnRNN:output_h:0*
T0*(
_output_shapes
:         а*
squeeze_dims
 s
	Squeeze_1SqueezeCudnnRNN:output_c:0*
T0*(
_output_shapes
:         а*
squeeze_dims
 R
ExpandDims_2/dimConst*
_output_shapes
: *
dtype0*
value	B :є
ExpandDims_2
ExpandDimsstrided_slice_0:output:0ExpandDims_2/dim:output:0*
T0*,
_output_shapes
:         а[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @a
IdentityIdentitystrided_slice_0:output:0*
T0*(
_output_shapes
:         аd

Identity_1IdentityExpandDims_2:output:0*
T0*,
_output_shapes
:         а[

Identity_2IdentitySqueeze:output:0*
T0*(
_output_shapes
:         а]

Identity_3IdentitySqueeze_1:output:0*
T0*(
_output_shapes
:         аI

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: "
concat_1concat_1_0:output:0"'
concat_1_axisconcat_1/axis:output:0"#
concat_axisconcat/axis:output:0"
cudnnrnnCudnnRNN:output_h:0"!

cudnnrnn_0CudnnRNN:output_c:0"

cudnnrnn_1CudnnRNN:output:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"!

expanddimsExpandDims:output:0"%
expanddims_1ExpandDims_1:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
init_cinit_c_0"
init_hinit_h_0"/
split_1_split_dimsplit_1/split_dim:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"+
split_split_dimsplit/split_dim:output:0")
strided_slicestrided_slice_0:output:0"
	transposetranspose_0:y:0"-
transpose_1_permtranspose_1/perm:output:0"-
transpose_2_permtranspose_2/perm:output:0"-
transpose_3_permtranspose_3/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"-
transpose_5_permtranspose_5/perm:output:0"-
transpose_6_permtranspose_6/perm:output:0"-
transpose_7_permtranspose_7/perm:output:0"-
transpose_8_permtranspose_8/perm:output:0")
transpose_permtranspose/perm:output:0*(
_construction_contextkEagerRuntime*p
_input_shapes_
]:         :         а:         а:	ђ	:
ађ	:ђ	*=
api_implements+)lstm_f0644406-b4c4-48ed-b733-8b83744a8936*
api_preferred_deviceGPU*Y
backward_function_name?=__inference___backward_gpu_lstm_with_fallback_1211903_1212079*
go_backwards( *

time_major( :S O
+
_output_shapes
:         
 
_user_specified_nameinputs:PL
(
_output_shapes
:         а
 
_user_specified_nameinit_h:PL
(
_output_shapes
:         а
 
_user_specified_nameinit_c:GC

_output_shapes
:	ђ	
 
_user_specified_namekernel:RN
 
_output_shapes
:
ађ	
*
_user_specified_namerecurrent_kernel:A=

_output_shapes	
:ђ	

_user_specified_namebias
╠	
э
E__inference_dense_13_layer_call_and_return_conditional_losses_1213883

inputs1
matmul_readvariableop_resource:	а-
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	а*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         а: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         а
 
_user_specified_nameinputs
№)
л
while_body_1211720
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_matmul_kernel_0%
!while_matmul_1_recurrent_kernel_0
while_biasadd_bias_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_matmul_kernel#
while_matmul_1_recurrent_kernel
while_biasadd_biasѕ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       д
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0њ
while/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while_matmul_kernel_0*
T0*(
_output_shapes
:         ђ	Ѓ
while/MatMul_1MatMulwhile_placeholder_2!while_matmul_1_recurrent_kernel_0*
T0*(
_output_shapes
:         ђ	w
	while/addAddV2while/MatMul:product:0while/MatMul_1:product:0*
T0*(
_output_shapes
:         ђ	p
while/BiasAddBiasAddwhile/add:z:0while_biasadd_bias_0*
T0*(
_output_shapes
:         ђ	W
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :╠
while/splitSplitwhile/split/split_dim:output:0while/BiasAdd:output:0*
T0*d
_output_shapesR
P:         а:         а:         а:         а*
	num_splita
while/SigmoidSigmoidwhile/split:output:0*
T0*(
_output_shapes
:         аc
while/Sigmoid_1Sigmoidwhile/split:output:1*
T0*(
_output_shapes
:         аm
	while/mulMulwhile/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:         а[

while/TanhTanhwhile/split:output:2*
T0*(
_output_shapes
:         аh
while/mul_1Mulwhile/Sigmoid:y:0while/Tanh:y:0*
T0*(
_output_shapes
:         аg
while/add_1AddV2while/mul:z:0while/mul_1:z:0*
T0*(
_output_shapes
:         аc
while/Sigmoid_2Sigmoidwhile/split:output:3*
T0*(
_output_shapes
:         аX
while/Tanh_1Tanhwhile/add_1:z:0*
T0*(
_output_shapes
:         аl
while/mul_2Mulwhile/Sigmoid_2:y:0while/Tanh_1:y:0*
T0*(
_output_shapes
:         аr
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : Я
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/mul_2:z:0*
_output_shapes
: *
element_dtype0:жУмO
while/add_2/yConst*
_output_shapes
: *
dtype0*
value	B :`
while/add_2AddV2while_placeholderwhile/add_2/y:output:0*
T0*
_output_shapes
: O
while/add_3/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_3AddV2while_while_loop_counterwhile/add_3/y:output:0*
T0*
_output_shapes
: L
while/IdentityIdentitywhile/add_3:z:0*
T0*
_output_shapes
: ]
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: N
while/Identity_2Identitywhile/add_2:z:0*
T0*
_output_shapes
: y
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: `
while/Identity_4Identitywhile/mul_2:z:0*
T0*(
_output_shapes
:         а`
while/Identity_5Identitywhile/add_1:z:0*
T0*(
_output_shapes
:         а"*
while_biasadd_biaswhile_biasadd_bias_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"D
while_matmul_1_recurrent_kernel!while_matmul_1_recurrent_kernel_0",
while_matmul_kernelwhile_matmul_kernel_0",
while_strided_slicewhile_strided_slice_0"е
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*e
_input_shapesT
R: : : : :         а:         а: : :	ђ	:
ађ	:ђ	: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:         а:.*
(
_output_shapes
:         а:

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	ђ	:&	"
 
_output_shapes
:
ађ	:!


_output_shapes	
:ђ	
њN
б
(__forward_gpu_lstm_with_fallback_1212999

inputs
init_h_0
init_c_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4
strided_slice
cudnnrnn

cudnnrnn_0

cudnnrnn_1

cudnnrnn_2
	transpose

expanddims
expanddims_1
concat_1
transpose_perm

init_h

init_c
concat_1_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
transpose_7_perm
transpose_8_perm
split_2_split_dim
split_split_dim
split_1_split_dim
concat_axisѕc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          B
transpose_0	Transposeinputstranspose/perm:output:0*
T0P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : r

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T0*,
_output_shapes
:         аR
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : v
ExpandDims_1
ExpandDimsinit_c_0ExpandDims_1/dim:output:0*
T0*,
_output_shapes
:         аQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ї
splitSplitsplit/split_dim:output:0kernel*
T0*@
_output_shapes.
,:	а:	а:	а:	а*
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ъ
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*D
_output_shapes2
0:
аа:
аа:
аа:
аа*
	num_splite
zeros_like/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB:ђ	U
zeros_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    x

zeros_likeFill#zeros_like/shape_as_tensor:output:0zeros_like/Const:output:0*
T0*
_output_shapes	
:ђ	M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : r
concatConcatV2zeros_like:output:0biasconcat/axis:output:0*
N*
T0*
_output_shapes	
:ђS
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : Ц
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*L
_output_shapes:
8:а:а:а:а:а:а:а:а*
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
         a
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       m
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes
:	аY
ReshapeReshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:аa
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       m
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes
:	а[
	Reshape_1Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:аa
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       m
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes
:	а[
	Reshape_2Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:аa
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       m
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0*
_output_shapes
:	а[
	Reshape_3Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes	
:аa
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       p
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0* 
_output_shapes
:
аа\
	Reshape_4Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes

:ђѕa
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       p
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0* 
_output_shapes
:
аа\
	Reshape_5Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes

:ђѕa
transpose_7/permConst*
_output_shapes
:*
dtype0*
valueB"       p
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0* 
_output_shapes
:
аа\
	Reshape_6Reshapetranspose_7:y:0Const:output:0*
T0*
_output_shapes

:ђѕa
transpose_8/permConst*
_output_shapes
:*
dtype0*
valueB"       p
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0* 
_output_shapes
:
аа\
	Reshape_7Reshapetranspose_8:y:0Const:output:0*
T0*
_output_shapes

:ђѕ\
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes	
:а\
	Reshape_9Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes	
:а]

Reshape_10Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes	
:а]

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes	
:а]

Reshape_12Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes	
:а]

Reshape_13Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes	
:а]

Reshape_14Reshapesplit_2:output:6Const:output:0*
T0*
_output_shapes	
:а]

Reshape_15Reshapesplit_2:output:7Const:output:0*
T0*
_output_shapes	
:аO
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ё

concat_1_0ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T0┌
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1_0:output:0*
T0*i
_output_shapesW
U:                  а:         а:         а:f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
         _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Й
strided_slice_0StridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
shrink_axis_maskq
SqueezeSqueezeCudnnRNN:output_h:0*
T0*(
_output_shapes
:         а*
squeeze_dims
 s
	Squeeze_1SqueezeCudnnRNN:output_c:0*
T0*(
_output_shapes
:         а*
squeeze_dims
 R
ExpandDims_2/dimConst*
_output_shapes
: *
dtype0*
value	B :є
ExpandDims_2
ExpandDimsstrided_slice_0:output:0ExpandDims_2/dim:output:0*
T0*,
_output_shapes
:         а[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @a
IdentityIdentitystrided_slice_0:output:0*
T0*(
_output_shapes
:         аd

Identity_1IdentityExpandDims_2:output:0*
T0*,
_output_shapes
:         а[

Identity_2IdentitySqueeze:output:0*
T0*(
_output_shapes
:         а]

Identity_3IdentitySqueeze_1:output:0*
T0*(
_output_shapes
:         аI

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: "
concat_1concat_1_0:output:0"'
concat_1_axisconcat_1/axis:output:0"#
concat_axisconcat/axis:output:0"
cudnnrnnCudnnRNN:output_h:0"!

cudnnrnn_0CudnnRNN:output_c:0"

cudnnrnn_1CudnnRNN:output:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"!

expanddimsExpandDims:output:0"%
expanddims_1ExpandDims_1:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
init_cinit_c_0"
init_hinit_h_0"/
split_1_split_dimsplit_1/split_dim:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"+
split_split_dimsplit/split_dim:output:0")
strided_slicestrided_slice_0:output:0"
	transposetranspose_0:y:0"-
transpose_1_permtranspose_1/perm:output:0"-
transpose_2_permtranspose_2/perm:output:0"-
transpose_3_permtranspose_3/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"-
transpose_5_permtranspose_5/perm:output:0"-
transpose_6_permtranspose_6/perm:output:0"-
transpose_7_permtranspose_7/perm:output:0"-
transpose_8_permtranspose_8/perm:output:0")
transpose_permtranspose/perm:output:0*(
_construction_contextkEagerRuntime*y
_input_shapesh
f:                  :         а:         а:	ђ	:
ађ	:ђ	*=
api_implements+)lstm_bc8f3e75-dc57-4569-96f0-8e80ed216f21*
api_preferred_deviceGPU*Y
backward_function_name?=__inference___backward_gpu_lstm_with_fallback_1212824_1213000*
go_backwards( *

time_major( :\ X
4
_output_shapes"
 :                  
 
_user_specified_nameinputs:PL
(
_output_shapes
:         а
 
_user_specified_nameinit_h:PL
(
_output_shapes
:         а
 
_user_specified_nameinit_c:GC

_output_shapes
:	ђ	
 
_user_specified_namekernel:RN
 
_output_shapes
:
ађ	
*
_user_specified_namerecurrent_kernel:A=

_output_shapes	
:ђ	

_user_specified_namebias
к;
┴
!__inference_standard_lstm_1209212

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:         B
ShapeShapetranspose:y:0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Л
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ▓
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмє
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       Я
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУм_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ж
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maske
MatMulMatMulstrided_slice_1:output:0kernel*
T0*(
_output_shapes
:         ђ	_
MatMul_1MatMulinit_hrecurrent_kernel*
T0*(
_output_shapes
:         ђ	e
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         ђ	T
BiasAddBiasAddadd:z:0bias*
T0*(
_output_shapes
:         ђ	Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :║
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:         а:         а:         а:         а*
	num_splitU
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:         аW
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:         аT
mulMulSigmoid_1:y:0init_c*
T0*(
_output_shapes
:         аO
TanhTanhsplit:output:2*
T0*(
_output_shapes
:         аV
mul_1MulSigmoid:y:0Tanh:y:0*
T0*(
_output_shapes
:         аU
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:         аW
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:         аL
Tanh_1Tanh	add_1:z:0*
T0*(
_output_shapes
:         аZ
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*(
_output_shapes
:         аn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :┼
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Й
whileStatelessWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hinit_cstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelrecurrent_kernelbias*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*f
_output_shapesT
R: : : : :         а:         а: : :	ђ	:
ађ	:ђ	* 
_read_only_resource_inputs
 *
_stateful_parallelism( *
bodyR
while_body_1209126*
condR
while_cond_1209125*e
output_shapesT
R: : : : :         а:         а: : :	ђ	:
ађ	:ђ	*
parallel_iterations Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       О
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:         а*
element_dtype0*
num_elementsh
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ѕ
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         а*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ќ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:         а[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  ђ?a
IdentityIdentitystrided_slice_2:output:0*
T0*(
_output_shapes
:         а^

Identity_1Identitytranspose_1:y:0*
T0*,
_output_shapes
:         аY

Identity_2Identitywhile:output:4*
T0*(
_output_shapes
:         аY

Identity_3Identitywhile:output:5*
T0*(
_output_shapes
:         аI

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*p
_input_shapes_
]:         :         а:         а:	ђ	:
ађ	:ђ	*=
api_implements+)lstm_4726b1d6-6e07-45d8-b9f1-158de8b4230f*
api_preferred_deviceCPU*
go_backwards( *

time_major( :S O
+
_output_shapes
:         
 
_user_specified_nameinputs:PL
(
_output_shapes
:         а
 
_user_specified_nameinit_h:PL
(
_output_shapes
:         а
 
_user_specified_nameinit_c:GC

_output_shapes
:	ђ	
 
_user_specified_namekernel:RN
 
_output_shapes
:
ађ	
*
_user_specified_namerecurrent_kernel:A=

_output_shapes	
:ђ	

_user_specified_namebias
к;
┴
!__inference_standard_lstm_1211360

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:         B
ShapeShapetranspose:y:0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Л
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ▓
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмє
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       Я
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУм_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ж
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maske
MatMulMatMulstrided_slice_1:output:0kernel*
T0*(
_output_shapes
:         ђ	_
MatMul_1MatMulinit_hrecurrent_kernel*
T0*(
_output_shapes
:         ђ	e
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         ђ	T
BiasAddBiasAddadd:z:0bias*
T0*(
_output_shapes
:         ђ	Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :║
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:         а:         а:         а:         а*
	num_splitU
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:         аW
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:         аT
mulMulSigmoid_1:y:0init_c*
T0*(
_output_shapes
:         аO
TanhTanhsplit:output:2*
T0*(
_output_shapes
:         аV
mul_1MulSigmoid:y:0Tanh:y:0*
T0*(
_output_shapes
:         аU
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:         аW
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:         аL
Tanh_1Tanh	add_1:z:0*
T0*(
_output_shapes
:         аZ
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*(
_output_shapes
:         аn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :┼
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Й
whileStatelessWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hinit_cstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelrecurrent_kernelbias*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*f
_output_shapesT
R: : : : :         а:         а: : :	ђ	:
ађ	:ђ	* 
_read_only_resource_inputs
 *
_stateful_parallelism( *
bodyR
while_body_1211274*
condR
while_cond_1211273*e
output_shapesT
R: : : : :         а:         а: : :	ђ	:
ађ	:ђ	*
parallel_iterations Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       О
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:         а*
element_dtype0*
num_elementsh
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ѕ
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         а*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ќ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:         а[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  ђ?a
IdentityIdentitystrided_slice_2:output:0*
T0*(
_output_shapes
:         а^

Identity_1Identitytranspose_1:y:0*
T0*,
_output_shapes
:         аY

Identity_2Identitywhile:output:4*
T0*(
_output_shapes
:         аY

Identity_3Identitywhile:output:5*
T0*(
_output_shapes
:         аI

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*p
_input_shapes_
]:         :         а:         а:	ђ	:
ађ	:ђ	*=
api_implements+)lstm_cbe182d0-608f-49e0-986e-6c0a38bbebc2*
api_preferred_deviceCPU*
go_backwards( *

time_major( :S O
+
_output_shapes
:         
 
_user_specified_nameinputs:PL
(
_output_shapes
:         а
 
_user_specified_nameinit_h:PL
(
_output_shapes
:         а
 
_user_specified_nameinit_c:GC

_output_shapes
:	ђ	
 
_user_specified_namekernel:RN
 
_output_shapes
:
ађ	
*
_user_specified_namerecurrent_kernel:A=

_output_shapes	
:ђ	

_user_specified_namebias
у
Ш
1__inference_change_layer_12_layer_call_fn_1210280

inputs
unknown:	ђ	
	unknown_0:
ађ	
	unknown_1:	ђ	
	unknown_2:	а
	unknown_3:
identityѕбStatefulPartitionedCallї
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         *'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8ѓ *U
fPRN
L__inference_change_layer_12_layer_call_and_return_conditional_losses_1210171s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:         : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
Ж
э
1__inference_change_layer_12_layer_call_fn_1210138
input_1
unknown:	ђ	
	unknown_0:
ађ	
	unknown_1:	ђ	
	unknown_2:	а
	unknown_3:
identityѕбStatefulPartitionedCallЇ
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         *'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8ѓ *U
fPRN
L__inference_change_layer_12_layer_call_and_return_conditional_losses_1210125s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:         : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:         
!
_user_specified_name	input_1
┴B
═
*__inference_gpu_lstm_with_fallback_1211456

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4ѕc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:         P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : p

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0*,
_output_shapes
:         аR
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : t
ExpandDims_1
ExpandDimsinit_cExpandDims_1/dim:output:0*
T0*,
_output_shapes
:         аQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ї
splitSplitsplit/split_dim:output:0kernel*
T0*@
_output_shapes.
,:	а:	а:	а:	а*
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ъ
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*D
_output_shapes2
0:
аа:
аа:
аа:
аа*
	num_splite
zeros_like/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB:ђ	U
zeros_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    x

zeros_likeFill#zeros_like/shape_as_tensor:output:0zeros_like/Const:output:0*
T0*
_output_shapes	
:ђ	M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : r
concatConcatV2zeros_like:output:0biasconcat/axis:output:0*
N*
T0*
_output_shapes	
:ђS
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : Ц
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*L
_output_shapes:
8:а:а:а:а:а:а:а:а*
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
         a
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       m
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes
:	аY
ReshapeReshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:аa
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       m
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes
:	а[
	Reshape_1Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:аa
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       m
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes
:	а[
	Reshape_2Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:аa
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       m
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0*
_output_shapes
:	а[
	Reshape_3Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes	
:аa
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       p
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0* 
_output_shapes
:
аа\
	Reshape_4Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes

:ђѕa
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       p
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0* 
_output_shapes
:
аа\
	Reshape_5Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes

:ђѕa
transpose_7/permConst*
_output_shapes
:*
dtype0*
valueB"       p
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0* 
_output_shapes
:
аа\
	Reshape_6Reshapetranspose_7:y:0Const:output:0*
T0*
_output_shapes

:ђѕa
transpose_8/permConst*
_output_shapes
:*
dtype0*
valueB"       p
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0* 
_output_shapes
:
аа\
	Reshape_7Reshapetranspose_8:y:0Const:output:0*
T0*
_output_shapes

:ђѕ\
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes	
:а\
	Reshape_9Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes	
:а]

Reshape_10Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes	
:а]

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes	
:а]

Reshape_12Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes	
:а]

Reshape_13Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes	
:а]

Reshape_14Reshapesplit_2:output:6Const:output:0*
T0*
_output_shapes	
:а]

Reshape_15Reshapesplit_2:output:7Const:output:0*
T0*
_output_shapes	
:аO
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : а
concat_1ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T0*
_output_shapes

:ђ╗═
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1:output:0*
T0*`
_output_shapesN
L:         а:         а:         а:f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
         _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Т
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         а*
shrink_axis_maskq
SqueezeSqueezeCudnnRNN:output_h:0*
T0*(
_output_shapes
:         а*
squeeze_dims
 s
	Squeeze_1SqueezeCudnnRNN:output_c:0*
T0*(
_output_shapes
:         а*
squeeze_dims
 R
ExpandDims_2/dimConst*
_output_shapes
: *
dtype0*
value	B :ё
ExpandDims_2
ExpandDimsstrided_slice:output:0ExpandDims_2/dim:output:0*
T0*,
_output_shapes
:         а[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @_
IdentityIdentitystrided_slice:output:0*
T0*(
_output_shapes
:         аd

Identity_1IdentityExpandDims_2:output:0*
T0*,
_output_shapes
:         а[

Identity_2IdentitySqueeze:output:0*
T0*(
_output_shapes
:         а]

Identity_3IdentitySqueeze_1:output:0*
T0*(
_output_shapes
:         аI

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*p
_input_shapes_
]:         :         а:         а:	ђ	:
ађ	:ђ	*=
api_implements+)lstm_cbe182d0-608f-49e0-986e-6c0a38bbebc2*
api_preferred_deviceGPU*
go_backwards( *

time_major( :S O
+
_output_shapes
:         
 
_user_specified_nameinputs:PL
(
_output_shapes
:         а
 
_user_specified_nameinit_h:PL
(
_output_shapes
:         а
 
_user_specified_nameinit_c:GC

_output_shapes
:	ђ	
 
_user_specified_namekernel:RN
 
_output_shapes
:
ађ	
*
_user_specified_namerecurrent_kernel:A=

_output_shapes	
:ђ	

_user_specified_namebias
┤
в
%__inference_signature_wrapper_1210250
input_1
unknown:	ђ	
	unknown_0:
ађ	
	unknown_1:	ђ	
	unknown_2:	а
	unknown_3:
identityѕбStatefulPartitionedCallс
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         *'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8ѓ *+
f&R$
"__inference__wrapped_model_1208164s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:         : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:         
!
_user_specified_name	input_1
╚
▄
J__inference_sequential_13_layer_call_and_return_conditional_losses_1209527

inputs"
lstm_13_1209488:	ђ	#
lstm_13_1209490:
ађ	
lstm_13_1209492:	ђ	#
dense_13_1209506:	а
dense_13_1209508:
identityѕб dense_13/StatefulPartitionedCallбlstm_13/StatefulPartitionedCallЃ
lstm_13/StatefulPartitionedCallStatefulPartitionedCallinputslstm_13_1209488lstm_13_1209490lstm_13_1209492*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         а*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_lstm_13_layer_call_and_return_conditional_losses_1209487Ћ
 dense_13/StatefulPartitionedCallStatefulPartitionedCall(lstm_13/StatefulPartitionedCall:output:0dense_13_1209506dense_13_1209508*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *N
fIRG
E__inference_dense_13_layer_call_and_return_conditional_losses_1209505С
reshape_13/PartitionedCallPartitionedCall)dense_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *P
fKRI
G__inference_reshape_13_layer_call_and_return_conditional_losses_1209524v
IdentityIdentity#reshape_13/PartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         І
NoOpNoOp!^dense_13/StatefulPartitionedCall ^lstm_13/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:         : : : : : 2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall2B
lstm_13/StatefulPartitionedCalllstm_13/StatefulPartitionedCall:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
П
с
J__inference_sequential_13_layer_call_and_return_conditional_losses_1210088
lstm_13_input"
lstm_13_1210074:	ђ	#
lstm_13_1210076:
ађ	
lstm_13_1210078:	ђ	#
dense_13_1210081:	а
dense_13_1210083:
identityѕб dense_13/StatefulPartitionedCallбlstm_13/StatefulPartitionedCallі
lstm_13/StatefulPartitionedCallStatefulPartitionedCalllstm_13_inputlstm_13_1210074lstm_13_1210076lstm_13_1210078*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         а*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_lstm_13_layer_call_and_return_conditional_losses_1209487Ћ
 dense_13/StatefulPartitionedCallStatefulPartitionedCall(lstm_13/StatefulPartitionedCall:output:0dense_13_1210081dense_13_1210083*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *N
fIRG
E__inference_dense_13_layer_call_and_return_conditional_losses_1209505С
reshape_13/PartitionedCallPartitionedCall)dense_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *P
fKRI
G__inference_reshape_13_layer_call_and_return_conditional_losses_1209524v
IdentityIdentity#reshape_13/PartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         І
NoOpNoOp!^dense_13/StatefulPartitionedCall ^lstm_13/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:         : : : : : 2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall2B
lstm_13/StatefulPartitionedCalllstm_13/StatefulPartitionedCall:Z V
+
_output_shapes
:         
'
_user_specified_namelstm_13_input
эM
б
(__forward_gpu_lstm_with_fallback_1208145

inputs
init_h_0
init_c_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4
strided_slice
cudnnrnn

cudnnrnn_0

cudnnrnn_1

cudnnrnn_2
	transpose

expanddims
expanddims_1
concat_1
transpose_perm

init_h

init_c
concat_1_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
transpose_7_perm
transpose_8_perm
split_2_split_dim
split_split_dim
split_1_split_dim
concat_axisѕc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          B
transpose_0	Transposeinputstranspose/perm:output:0*
T0P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : r

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T0*,
_output_shapes
:         аR
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : v
ExpandDims_1
ExpandDimsinit_c_0ExpandDims_1/dim:output:0*
T0*,
_output_shapes
:         аQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ї
splitSplitsplit/split_dim:output:0kernel*
T0*@
_output_shapes.
,:	а:	а:	а:	а*
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ъ
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*D
_output_shapes2
0:
аа:
аа:
аа:
аа*
	num_splite
zeros_like/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB:ђ	U
zeros_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    x

zeros_likeFill#zeros_like/shape_as_tensor:output:0zeros_like/Const:output:0*
T0*
_output_shapes	
:ђ	M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : r
concatConcatV2zeros_like:output:0biasconcat/axis:output:0*
N*
T0*
_output_shapes	
:ђS
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : Ц
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*L
_output_shapes:
8:а:а:а:а:а:а:а:а*
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
         a
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       m
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes
:	аY
ReshapeReshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:аa
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       m
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes
:	а[
	Reshape_1Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:аa
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       m
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes
:	а[
	Reshape_2Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:аa
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       m
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0*
_output_shapes
:	а[
	Reshape_3Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes	
:аa
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       p
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0* 
_output_shapes
:
аа\
	Reshape_4Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes

:ђѕa
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       p
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0* 
_output_shapes
:
аа\
	Reshape_5Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes

:ђѕa
transpose_7/permConst*
_output_shapes
:*
dtype0*
valueB"       p
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0* 
_output_shapes
:
аа\
	Reshape_6Reshapetranspose_7:y:0Const:output:0*
T0*
_output_shapes

:ђѕa
transpose_8/permConst*
_output_shapes
:*
dtype0*
valueB"       p
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0* 
_output_shapes
:
аа\
	Reshape_7Reshapetranspose_8:y:0Const:output:0*
T0*
_output_shapes

:ђѕ\
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes	
:а\
	Reshape_9Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes	
:а]

Reshape_10Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes	
:а]

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes	
:а]

Reshape_12Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes	
:а]

Reshape_13Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes	
:а]

Reshape_14Reshapesplit_2:output:6Const:output:0*
T0*
_output_shapes	
:а]

Reshape_15Reshapesplit_2:output:7Const:output:0*
T0*
_output_shapes	
:аO
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ё

concat_1_0ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T0Л
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1_0:output:0*
T0*`
_output_shapesN
L:         а:         а:         а:f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
         _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Й
strided_slice_0StridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
shrink_axis_maskq
SqueezeSqueezeCudnnRNN:output_h:0*
T0*(
_output_shapes
:         а*
squeeze_dims
 s
	Squeeze_1SqueezeCudnnRNN:output_c:0*
T0*(
_output_shapes
:         а*
squeeze_dims
 R
ExpandDims_2/dimConst*
_output_shapes
: *
dtype0*
value	B :є
ExpandDims_2
ExpandDimsstrided_slice_0:output:0ExpandDims_2/dim:output:0*
T0*,
_output_shapes
:         а[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @a
IdentityIdentitystrided_slice_0:output:0*
T0*(
_output_shapes
:         аd

Identity_1IdentityExpandDims_2:output:0*
T0*,
_output_shapes
:         а[

Identity_2IdentitySqueeze:output:0*
T0*(
_output_shapes
:         а]

Identity_3IdentitySqueeze_1:output:0*
T0*(
_output_shapes
:         аI

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: "
concat_1concat_1_0:output:0"'
concat_1_axisconcat_1/axis:output:0"#
concat_axisconcat/axis:output:0"
cudnnrnnCudnnRNN:output_h:0"!

cudnnrnn_0CudnnRNN:output_c:0"

cudnnrnn_1CudnnRNN:output:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"!

expanddimsExpandDims:output:0"%
expanddims_1ExpandDims_1:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
init_cinit_c_0"
init_hinit_h_0"/
split_1_split_dimsplit_1/split_dim:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"+
split_split_dimsplit/split_dim:output:0")
strided_slicestrided_slice_0:output:0"
	transposetranspose_0:y:0"-
transpose_1_permtranspose_1/perm:output:0"-
transpose_2_permtranspose_2/perm:output:0"-
transpose_3_permtranspose_3/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"-
transpose_5_permtranspose_5/perm:output:0"-
transpose_6_permtranspose_6/perm:output:0"-
transpose_7_permtranspose_7/perm:output:0"-
transpose_8_permtranspose_8/perm:output:0")
transpose_permtranspose/perm:output:0*(
_construction_contextkEagerRuntime*p
_input_shapes_
]:         :         а:         а:	ђ	:
ађ	:ђ	*=
api_implements+)lstm_e15ea421-5ae9-42eb-bcec-970c1aedc419*
api_preferred_deviceGPU*Y
backward_function_name?=__inference___backward_gpu_lstm_with_fallback_1207970_1208146*
go_backwards( *

time_major( :S O
+
_output_shapes
:         
 
_user_specified_nameinputs:PL
(
_output_shapes
:         а
 
_user_specified_nameinit_h:PL
(
_output_shapes
:         а
 
_user_specified_nameinit_c:GC

_output_shapes
:	ђ	
 
_user_specified_namekernel:RN
 
_output_shapes
:
ађ	
*
_user_specified_namerecurrent_kernel:A=

_output_shapes	
:ђ	

_user_specified_namebias
ф┬
у
=__inference___backward_gpu_lstm_with_fallback_1209822_1209998
placeholder
placeholder_1
placeholder_2
placeholder_3
placeholder_43
/gradients_expanddims_2_grad_shape_strided_slice)
%gradients_squeeze_grad_shape_cudnnrnn+
'gradients_squeeze_1_grad_shape_cudnnrnn/
+gradients_strided_slice_grad_shape_cudnnrnn!
gradients_zeros_like_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9
5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_15
1gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h,
(gradients_expanddims_1_grad_shape_init_c-
)gradients_concat_1_grad_mod_concat_1_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_permA
=gradients_transpose_7_grad_invertpermutation_transpose_7_permA
=gradients_transpose_8_grad_invertpermutation_transpose_8_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim)
%gradients_concat_grad_mod_concat_axis
identity

identity_1

identity_2

identity_3

identity_4

identity_5ѕ_
gradients/grad_ys_0Identityplaceholder*
T0*(
_output_shapes
:         аe
gradients/grad_ys_1Identityplaceholder_1*
T0*,
_output_shapes
:         аa
gradients/grad_ys_2Identityplaceholder_2*
T0*(
_output_shapes
:         аa
gradients/grad_ys_3Identityplaceholder_3*
T0*(
_output_shapes
:         аO
gradients/grad_ys_4Identityplaceholder_4*
T0*
_output_shapes
: ђ
!gradients/ExpandDims_2_grad/ShapeShape/gradients_expanddims_2_grad_shape_strided_slice*
T0*
_output_shapes
:Ф
#gradients/ExpandDims_2_grad/ReshapeReshapegradients/grad_ys_1:output:0*gradients/ExpandDims_2_grad/Shape:output:0*
T0*(
_output_shapes
:         аq
gradients/Squeeze_grad/ShapeShape%gradients_squeeze_grad_shape_cudnnrnn*
T0*
_output_shapes
:Ц
gradients/Squeeze_grad/ReshapeReshapegradients/grad_ys_2:output:0%gradients/Squeeze_grad/Shape:output:0*
T0*,
_output_shapes
:         аu
gradients/Squeeze_1_grad/ShapeShape'gradients_squeeze_1_grad_shape_cudnnrnn*
T0*
_output_shapes
:Е
 gradients/Squeeze_1_grad/ReshapeReshapegradients/grad_ys_3:output:0'gradients/Squeeze_1_grad/Shape:output:0*
T0*,
_output_shapes
:         ак
gradients/AddNAddNgradients/grad_ys_0:output:0,gradients/ExpandDims_2_grad/Reshape:output:0*
N*
T0*&
_class
loc:@gradients/grad_ys_0*(
_output_shapes
:         а}
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
:є
3gradients/strided_slice_grad/StridedSliceGrad/beginConst*
_output_shapes
:*
dtype0*
valueB:
         {
1gradients/strided_slice_grad/StridedSliceGrad/endConst*
_output_shapes
:*
dtype0*
valueB: 
5gradients/strided_slice_grad/StridedSliceGrad/stridesConst*
_output_shapes
:*
dtype0*
valueB:џ
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0<gradients/strided_slice_grad/StridedSliceGrad/begin:output:0:gradients/strided_slice_grad/StridedSliceGrad/end:output:0>gradients/strided_slice_grad/StridedSliceGrad/strides:output:0gradients/AddN:sum:0*
Index0*
T0*,
_output_shapes
:         а*
shrink_axis_maskc
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
:Ц
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_11gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1+gradients_strided_slice_grad_shape_cudnnrnn%gradients_squeeze_grad_shape_cudnnrnn'gradients_squeeze_1_grad_shape_cudnnrnn6gradients/strided_slice_grad/StridedSliceGrad:output:0'gradients/Squeeze_grad/Reshape:output:0)gradients/Squeeze_1_grad/Reshape:output:0gradients_zeros_like_cudnnrnn*
T0*c
_output_shapesQ
O:         :         а:         а:ђ╗ќ
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:л
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*+
_output_shapes
:         u
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
:к
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*(
_output_shapes
:         аy
!gradients/ExpandDims_1_grad/ShapeShape(gradients_expanddims_1_grad_shape_init_c*
T0*
_output_shapes
:╩
#gradients/ExpandDims_1_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_c_backprop:0*gradients/ExpandDims_1_grad/Shape:output:0*
T0*(
_output_shapes
:         а^
gradients/concat_1_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :џ
gradients/concat_1_grad/modFloorMod)gradients_concat_1_grad_mod_concat_1_axis%gradients/concat_1_grad/Rank:output:0*
T0*
_output_shapes
: h
gradients/concat_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:аj
gradients/concat_1_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:аj
gradients/concat_1_grad/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:аj
gradients/concat_1_grad/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:аk
gradients/concat_1_grad/Shape_4Const*
_output_shapes
:*
dtype0*
valueB:ђѕk
gradients/concat_1_grad/Shape_5Const*
_output_shapes
:*
dtype0*
valueB:ђѕk
gradients/concat_1_grad/Shape_6Const*
_output_shapes
:*
dtype0*
valueB:ђѕk
gradients/concat_1_grad/Shape_7Const*
_output_shapes
:*
dtype0*
valueB:ђѕj
gradients/concat_1_grad/Shape_8Const*
_output_shapes
:*
dtype0*
valueB:аj
gradients/concat_1_grad/Shape_9Const*
_output_shapes
:*
dtype0*
valueB:аk
 gradients/concat_1_grad/Shape_10Const*
_output_shapes
:*
dtype0*
valueB:аk
 gradients/concat_1_grad/Shape_11Const*
_output_shapes
:*
dtype0*
valueB:аk
 gradients/concat_1_grad/Shape_12Const*
_output_shapes
:*
dtype0*
valueB:аk
 gradients/concat_1_grad/Shape_13Const*
_output_shapes
:*
dtype0*
valueB:аk
 gradients/concat_1_grad/Shape_14Const*
_output_shapes
:*
dtype0*
valueB:аk
 gradients/concat_1_grad/Shape_15Const*
_output_shapes
:*
dtype0*
valueB:аЭ
$gradients/concat_1_grad/ConcatOffsetConcatOffsetgradients/concat_1_grad/mod:z:0&gradients/concat_1_grad/Shape:output:0(gradients/concat_1_grad/Shape_1:output:0(gradients/concat_1_grad/Shape_2:output:0(gradients/concat_1_grad/Shape_3:output:0(gradients/concat_1_grad/Shape_4:output:0(gradients/concat_1_grad/Shape_5:output:0(gradients/concat_1_grad/Shape_6:output:0(gradients/concat_1_grad/Shape_7:output:0(gradients/concat_1_grad/Shape_8:output:0(gradients/concat_1_grad/Shape_9:output:0)gradients/concat_1_grad/Shape_10:output:0)gradients/concat_1_grad/Shape_11:output:0)gradients/concat_1_grad/Shape_12:output:0)gradients/concat_1_grad/Shape_13:output:0)gradients/concat_1_grad/Shape_14:output:0)gradients/concat_1_grad/Shape_15:output:0*
N*t
_output_shapesb
`::::::::::::::::В
gradients/concat_1_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:0&gradients/concat_1_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:а­
gradients/concat_1_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:1(gradients/concat_1_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:а­
gradients/concat_1_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:2(gradients/concat_1_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes	
:а­
gradients/concat_1_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:3(gradients/concat_1_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes	
:аы
gradients/concat_1_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:4(gradients/concat_1_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes

:ђѕы
gradients/concat_1_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:5(gradients/concat_1_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes

:ђѕы
gradients/concat_1_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:6(gradients/concat_1_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes

:ђѕы
gradients/concat_1_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:7(gradients/concat_1_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes

:ђѕ­
gradients/concat_1_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:8(gradients/concat_1_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes	
:а­
gradients/concat_1_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:9(gradients/concat_1_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes	
:аз
 gradients/concat_1_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:10)gradients/concat_1_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes	
:аз
 gradients/concat_1_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:11)gradients/concat_1_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes	
:аз
 gradients/concat_1_grad/Slice_12Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:12)gradients/concat_1_grad/Shape_12:output:0*
Index0*
T0*
_output_shapes	
:аз
 gradients/concat_1_grad/Slice_13Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:13)gradients/concat_1_grad/Shape_13:output:0*
Index0*
T0*
_output_shapes	
:аз
 gradients/concat_1_grad/Slice_14Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:14)gradients/concat_1_grad/Shape_14:output:0*
Index0*
T0*
_output_shapes	
:аз
 gradients/concat_1_grad/Slice_15Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:15)gradients/concat_1_grad/Shape_15:output:0*
Index0*
T0*
_output_shapes	
:аm
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      б
gradients/Reshape_grad/ReshapeReshape&gradients/concat_1_grad/Slice:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes
:	аo
gradients/Reshape_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      е
 gradients/Reshape_1_grad/ReshapeReshape(gradients/concat_1_grad/Slice_1:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0*
_output_shapes
:	аo
gradients/Reshape_2_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      е
 gradients/Reshape_2_grad/ReshapeReshape(gradients/concat_1_grad/Slice_2:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0*
_output_shapes
:	аo
gradients/Reshape_3_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      е
 gradients/Reshape_3_grad/ReshapeReshape(gradients/concat_1_grad/Slice_3:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0*
_output_shapes
:	аo
gradients/Reshape_4_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      Е
 gradients/Reshape_4_grad/ReshapeReshape(gradients/concat_1_grad/Slice_4:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0* 
_output_shapes
:
ааo
gradients/Reshape_5_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      Е
 gradients/Reshape_5_grad/ReshapeReshape(gradients/concat_1_grad/Slice_5:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0* 
_output_shapes
:
ааo
gradients/Reshape_6_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      Е
 gradients/Reshape_6_grad/ReshapeReshape(gradients/concat_1_grad/Slice_6:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0* 
_output_shapes
:
ааo
gradients/Reshape_7_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      Е
 gradients/Reshape_7_grad/ReshapeReshape(gradients/concat_1_grad/Slice_7:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0* 
_output_shapes
:
ааi
gradients/Reshape_8_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:ац
 gradients/Reshape_8_grad/ReshapeReshape(gradients/concat_1_grad/Slice_8:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes	
:аi
gradients/Reshape_9_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:ац
 gradients/Reshape_9_grad/ReshapeReshape(gradients/concat_1_grad/Slice_9:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes	
:аj
gradients/Reshape_10_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:аД
!gradients/Reshape_10_grad/ReshapeReshape)gradients/concat_1_grad/Slice_10:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes	
:аj
gradients/Reshape_11_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:аД
!gradients/Reshape_11_grad/ReshapeReshape)gradients/concat_1_grad/Slice_11:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes	
:аj
gradients/Reshape_12_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:аД
!gradients/Reshape_12_grad/ReshapeReshape)gradients/concat_1_grad/Slice_12:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes	
:аj
gradients/Reshape_13_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:аД
!gradients/Reshape_13_grad/ReshapeReshape)gradients/concat_1_grad/Slice_13:output:0(gradients/Reshape_13_grad/Shape:output:0*
T0*
_output_shapes	
:аj
gradients/Reshape_14_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:аД
!gradients/Reshape_14_grad/ReshapeReshape)gradients/concat_1_grad/Slice_14:output:0(gradients/Reshape_14_grad/Shape:output:0*
T0*
_output_shapes	
:аj
gradients/Reshape_15_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:аД
!gradients/Reshape_15_grad/ReshapeReshape)gradients/concat_1_grad/Slice_15:output:0(gradients/Reshape_15_grad/Shape:output:0*
T0*
_output_shapes	
:аю
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:Х
$gradients/transpose_1_grad/transpose	Transpose'gradients/Reshape_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0*
_output_shapes
:	аю
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:И
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0*
_output_shapes
:	аю
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:И
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0*
_output_shapes
:	аю
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:И
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0*
_output_shapes
:	аю
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:╣
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
ааю
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:╣
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
ааю
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:╣
$gradients/transpose_7_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
ааю
,gradients/transpose_8_grad/InvertPermutationInvertPermutation=gradients_transpose_8_grad_invertpermutation_transpose_8_perm*
_output_shapes
:╣
$gradients/transpose_8_grad/transpose	Transpose)gradients/Reshape_7_grad/Reshape:output:00gradients/transpose_8_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
аау
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0*gradients/Reshape_13_grad/Reshape:output:0*gradients/Reshape_14_grad/Reshape:output:0*gradients/Reshape_15_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
N*
T0*
_output_shapes	
:ђ»
gradients/split_grad/concatConcatV2(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
N*
T0*
_output_shapes
:	ђ	Х
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0(gradients/transpose_7_grad/transpose:y:0(gradients/transpose_8_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
N*
T0* 
_output_shapes
:
ађ	\
gradients/concat_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :њ
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: f
gradients/concat_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:ђ	h
gradients/concat_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:ђ	╩
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0*
N* 
_output_shapes
::м
gradients/concat_grad/SliceSlice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:ђ	о
gradients/concat_grad/Slice_1Slice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:ђ	r
IdentityIdentity&gradients/transpose_grad/transpose:y:0*
T0*+
_output_shapes
:         u

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0*
T0*(
_output_shapes
:         аw

Identity_2Identity,gradients/ExpandDims_1_grad/Reshape:output:0*
T0*(
_output_shapes
:         аf

Identity_3Identity$gradients/split_grad/concat:output:0*
T0*
_output_shapes
:	ђ	i

Identity_4Identity&gradients/split_1_grad/concat:output:0*
T0* 
_output_shapes
:
ађ	d

Identity_5Identity&gradients/concat_grad/Slice_1:output:0*
T0*
_output_shapes	
:ђ	"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*(
_construction_contextkEagerRuntime*ѓ
_input_shapes­
ь:         а:         а:         а:         а: :         а:         а:         а:         а::         :         а:         а:ђ╗::         а:         а: ::::::::: : : : *=
api_implements+)lstm_6b3f42d6-117f-4a93-968a-eb0630b50933*
api_preferred_deviceGPU*C
forward_function_name*(__forward_gpu_lstm_with_fallback_1209997*
go_backwards( *

time_major( :. *
(
_output_shapes
:         а:2.
,
_output_shapes
:         а:.*
(
_output_shapes
:         а:.*
(
_output_shapes
:         а:

_output_shapes
: :.*
(
_output_shapes
:         а:2.
,
_output_shapes
:         а:2.
,
_output_shapes
:         а:2.
,
_output_shapes
:         а:	

_output_shapes
::1
-
+
_output_shapes
:         :2.
,
_output_shapes
:         а:2.
,
_output_shapes
:         а:"

_output_shapes

:ђ╗: 

_output_shapes
::.*
(
_output_shapes
:         а:.*
(
_output_shapes
:         а:

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
њN
б
(__forward_gpu_lstm_with_fallback_1209038

inputs
init_h_0
init_c_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4
strided_slice
cudnnrnn

cudnnrnn_0

cudnnrnn_1

cudnnrnn_2
	transpose

expanddims
expanddims_1
concat_1
transpose_perm

init_h

init_c
concat_1_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
transpose_7_perm
transpose_8_perm
split_2_split_dim
split_split_dim
split_1_split_dim
concat_axisѕc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          B
transpose_0	Transposeinputstranspose/perm:output:0*
T0P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : r

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T0*,
_output_shapes
:         аR
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : v
ExpandDims_1
ExpandDimsinit_c_0ExpandDims_1/dim:output:0*
T0*,
_output_shapes
:         аQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ї
splitSplitsplit/split_dim:output:0kernel*
T0*@
_output_shapes.
,:	а:	а:	а:	а*
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ъ
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*D
_output_shapes2
0:
аа:
аа:
аа:
аа*
	num_splite
zeros_like/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB:ђ	U
zeros_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    x

zeros_likeFill#zeros_like/shape_as_tensor:output:0zeros_like/Const:output:0*
T0*
_output_shapes	
:ђ	M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : r
concatConcatV2zeros_like:output:0biasconcat/axis:output:0*
N*
T0*
_output_shapes	
:ђS
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : Ц
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*L
_output_shapes:
8:а:а:а:а:а:а:а:а*
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
         a
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       m
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes
:	аY
ReshapeReshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:аa
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       m
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes
:	а[
	Reshape_1Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:аa
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       m
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes
:	а[
	Reshape_2Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:аa
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       m
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0*
_output_shapes
:	а[
	Reshape_3Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes	
:аa
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       p
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0* 
_output_shapes
:
аа\
	Reshape_4Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes

:ђѕa
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       p
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0* 
_output_shapes
:
аа\
	Reshape_5Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes

:ђѕa
transpose_7/permConst*
_output_shapes
:*
dtype0*
valueB"       p
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0* 
_output_shapes
:
аа\
	Reshape_6Reshapetranspose_7:y:0Const:output:0*
T0*
_output_shapes

:ђѕa
transpose_8/permConst*
_output_shapes
:*
dtype0*
valueB"       p
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0* 
_output_shapes
:
аа\
	Reshape_7Reshapetranspose_8:y:0Const:output:0*
T0*
_output_shapes

:ђѕ\
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes	
:а\
	Reshape_9Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes	
:а]

Reshape_10Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes	
:а]

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes	
:а]

Reshape_12Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes	
:а]

Reshape_13Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes	
:а]

Reshape_14Reshapesplit_2:output:6Const:output:0*
T0*
_output_shapes	
:а]

Reshape_15Reshapesplit_2:output:7Const:output:0*
T0*
_output_shapes	
:аO
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ё

concat_1_0ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T0┌
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1_0:output:0*
T0*i
_output_shapesW
U:                  а:         а:         а:f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
         _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Й
strided_slice_0StridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
shrink_axis_maskq
SqueezeSqueezeCudnnRNN:output_h:0*
T0*(
_output_shapes
:         а*
squeeze_dims
 s
	Squeeze_1SqueezeCudnnRNN:output_c:0*
T0*(
_output_shapes
:         а*
squeeze_dims
 R
ExpandDims_2/dimConst*
_output_shapes
: *
dtype0*
value	B :є
ExpandDims_2
ExpandDimsstrided_slice_0:output:0ExpandDims_2/dim:output:0*
T0*,
_output_shapes
:         а[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @a
IdentityIdentitystrided_slice_0:output:0*
T0*(
_output_shapes
:         аd

Identity_1IdentityExpandDims_2:output:0*
T0*,
_output_shapes
:         а[

Identity_2IdentitySqueeze:output:0*
T0*(
_output_shapes
:         а]

Identity_3IdentitySqueeze_1:output:0*
T0*(
_output_shapes
:         аI

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: "
concat_1concat_1_0:output:0"'
concat_1_axisconcat_1/axis:output:0"#
concat_axisconcat/axis:output:0"
cudnnrnnCudnnRNN:output_h:0"!

cudnnrnn_0CudnnRNN:output_c:0"

cudnnrnn_1CudnnRNN:output:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"!

expanddimsExpandDims:output:0"%
expanddims_1ExpandDims_1:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
init_cinit_c_0"
init_hinit_h_0"/
split_1_split_dimsplit_1/split_dim:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"+
split_split_dimsplit/split_dim:output:0")
strided_slicestrided_slice_0:output:0"
	transposetranspose_0:y:0"-
transpose_1_permtranspose_1/perm:output:0"-
transpose_2_permtranspose_2/perm:output:0"-
transpose_3_permtranspose_3/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"-
transpose_5_permtranspose_5/perm:output:0"-
transpose_6_permtranspose_6/perm:output:0"-
transpose_7_permtranspose_7/perm:output:0"-
transpose_8_permtranspose_8/perm:output:0")
transpose_permtranspose/perm:output:0*(
_construction_contextkEagerRuntime*y
_input_shapesh
f:                  :         а:         а:	ђ	:
ађ	:ђ	*=
api_implements+)lstm_8b39bb2f-e1e5-4618-9947-ffe4540afe4b*
api_preferred_deviceGPU*Y
backward_function_name?=__inference___backward_gpu_lstm_with_fallback_1208863_1209039*
go_backwards( *

time_major( :\ X
4
_output_shapes"
 :                  
 
_user_specified_nameinputs:PL
(
_output_shapes
:         а
 
_user_specified_nameinit_h:PL
(
_output_shapes
:         а
 
_user_specified_nameinit_c:GC

_output_shapes
:	ђ	
 
_user_specified_namekernel:RN
 
_output_shapes
:
ађ	
*
_user_specified_namerecurrent_kernel:A=

_output_shapes	
:ђ	

_user_specified_namebias
й
┴
D__inference_lstm_13_layer_call_and_return_conditional_losses_1213002
inputs_0/
read_readvariableop_resource:	ђ	2
read_1_readvariableop_resource:
ађ	-
read_2_readvariableop_resource:	ђ	

identity_3ѕбRead/ReadVariableOpбRead_1/ReadVariableOpбRead_2/ReadVariableOp=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Л
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :аs
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:         аS
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :аw
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    s
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:         аq
Read/ReadVariableOpReadVariableOpread_readvariableop_resource*
_output_shapes
:	ђ	*
dtype0[
IdentityIdentityRead/ReadVariableOp:value:0*
T0*
_output_shapes
:	ђ	v
Read_1/ReadVariableOpReadVariableOpread_1_readvariableop_resource* 
_output_shapes
:
ађ	*
dtype0`

Identity_1IdentityRead_1/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ађ	q
Read_2/ReadVariableOpReadVariableOpread_2_readvariableop_resource*
_output_shapes	
:ђ	*
dtype0[

Identity_2IdentityRead_2/ReadVariableOp:value:0*
T0*
_output_shapes	
:ђ	╣
PartitionedCallPartitionedCallinputs_0zeros:output:0zeros_1:output:0Identity:output:0Identity_1:output:0Identity_2:output:0*
Tin

2*
Tout	
2*
_collective_manager_ids
 *j
_output_shapesX
V:         а:         а:         а:         а: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ **
f%R#
!__inference_standard_lstm_1212727j

Identity_3IdentityPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         аї
NoOpNoOp^Read/ReadVariableOp^Read_1/ReadVariableOp^Read_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  : : : 2*
Read/ReadVariableOpRead/ReadVariableOp2.
Read_1/ReadVariableOpRead_1/ReadVariableOp2.
Read_2/ReadVariableOpRead_2/ReadVariableOp:^ Z
4
_output_shapes"
 :                  
"
_user_specified_name
inputs_0
р;
┴
!__inference_standard_lstm_1208766

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :                  B
ShapeShapetranspose:y:0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Л
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ▓
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмє
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       Я
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУм_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ж
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maske
MatMulMatMulstrided_slice_1:output:0kernel*
T0*(
_output_shapes
:         ђ	_
MatMul_1MatMulinit_hrecurrent_kernel*
T0*(
_output_shapes
:         ђ	e
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         ђ	T
BiasAddBiasAddadd:z:0bias*
T0*(
_output_shapes
:         ђ	Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :║
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:         а:         а:         а:         а*
	num_splitU
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:         аW
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:         аT
mulMulSigmoid_1:y:0init_c*
T0*(
_output_shapes
:         аO
TanhTanhsplit:output:2*
T0*(
_output_shapes
:         аV
mul_1MulSigmoid:y:0Tanh:y:0*
T0*(
_output_shapes
:         аU
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:         аW
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:         аL
Tanh_1Tanh	add_1:z:0*
T0*(
_output_shapes
:         аZ
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*(
_output_shapes
:         аn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :┼
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Й
whileStatelessWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hinit_cstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelrecurrent_kernelbias*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*f
_output_shapesT
R: : : : :         а:         а: : :	ђ	:
ађ	:ђ	* 
_read_only_resource_inputs
 *
_stateful_parallelism( *
bodyR
while_body_1208680*
condR
while_cond_1208679*e
output_shapesT
R: : : : :         а:         а: : :	ђ	:
ађ	:ђ	*
parallel_iterations Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       О
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:         а*
element_dtype0*
num_elementsh
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ѕ
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         а*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ќ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:         а[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  ђ?a
IdentityIdentitystrided_slice_2:output:0*
T0*(
_output_shapes
:         а^

Identity_1Identitytranspose_1:y:0*
T0*,
_output_shapes
:         аY

Identity_2Identitywhile:output:4*
T0*(
_output_shapes
:         аY

Identity_3Identitywhile:output:5*
T0*(
_output_shapes
:         аI

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*y
_input_shapesh
f:                  :         а:         а:	ђ	:
ађ	:ђ	*=
api_implements+)lstm_8b39bb2f-e1e5-4618-9947-ffe4540afe4b*
api_preferred_deviceCPU*
go_backwards( *

time_major( :\ X
4
_output_shapes"
 :                  
 
_user_specified_nameinputs:PL
(
_output_shapes
:         а
 
_user_specified_nameinit_h:PL
(
_output_shapes
:         а
 
_user_specified_nameinit_c:GC

_output_shapes
:	ђ	
 
_user_specified_namekernel:RN
 
_output_shapes
:
ађ	
*
_user_specified_namerecurrent_kernel:A=

_output_shapes	
:ђ	

_user_specified_namebias
П
с
J__inference_sequential_13_layer_call_and_return_conditional_losses_1210105
lstm_13_input"
lstm_13_1210091:	ђ	#
lstm_13_1210093:
ађ	
lstm_13_1210095:	ђ	#
dense_13_1210098:	а
dense_13_1210100:
identityѕб dense_13/StatefulPartitionedCallбlstm_13/StatefulPartitionedCallі
lstm_13/StatefulPartitionedCallStatefulPartitionedCalllstm_13_inputlstm_13_1210091lstm_13_1210093lstm_13_1210095*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         а*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_lstm_13_layer_call_and_return_conditional_losses_1210000Ћ
 dense_13/StatefulPartitionedCallStatefulPartitionedCall(lstm_13/StatefulPartitionedCall:output:0dense_13_1210098dense_13_1210100*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *N
fIRG
E__inference_dense_13_layer_call_and_return_conditional_losses_1209505С
reshape_13/PartitionedCallPartitionedCall)dense_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *P
fKRI
G__inference_reshape_13_layer_call_and_return_conditional_losses_1209524v
IdentityIdentity#reshape_13/PartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         І
NoOpNoOp!^dense_13/StatefulPartitionedCall ^lstm_13/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:         : : : : : 2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall2B
lstm_13/StatefulPartitionedCalllstm_13/StatefulPartitionedCall:Z V
+
_output_shapes
:         
'
_user_specified_namelstm_13_input
Ч
и
)__inference_lstm_13_layer_call_fn_1212140

inputs
unknown:	ђ	
	unknown_0:
ађ	
	unknown_1:	ђ	
identityѕбStatefulPartitionedCallу
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         а*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_lstm_13_layer_call_and_return_conditional_losses_1210000p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         а`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
Е
я
L__inference_change_layer_12_layer_call_and_return_conditional_losses_1210215
input_1(
sequential_13_1210202:	ђ	)
sequential_13_1210204:
ађ	$
sequential_13_1210206:	ђ	(
sequential_13_1210208:	а#
sequential_13_1210210:
identityѕб%sequential_13/StatefulPartitionedCallО
%sequential_13/StatefulPartitionedCallStatefulPartitionedCallinput_1sequential_13_1210202sequential_13_1210204sequential_13_1210206sequential_13_1210208sequential_13_1210210*
Tin

2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         *'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8ѓ *S
fNRL
J__inference_sequential_13_layer_call_and_return_conditional_losses_1209527{
addAddV2input_1.sequential_13/StatefulPartitionedCall:output:0*
T0*+
_output_shapes
:         Z
IdentityIdentityadd:z:0^NoOp*
T0*+
_output_shapes
:         n
NoOpNoOp&^sequential_13/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:         : : : : : 2N
%sequential_13/StatefulPartitionedCall%sequential_13/StatefulPartitionedCall:T P
+
_output_shapes
:         
!
_user_specified_name	input_1
ф┬
у
=__inference___backward_gpu_lstm_with_fallback_1210980_1211156
placeholder
placeholder_1
placeholder_2
placeholder_3
placeholder_43
/gradients_expanddims_2_grad_shape_strided_slice)
%gradients_squeeze_grad_shape_cudnnrnn+
'gradients_squeeze_1_grad_shape_cudnnrnn/
+gradients_strided_slice_grad_shape_cudnnrnn!
gradients_zeros_like_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9
5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_15
1gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h,
(gradients_expanddims_1_grad_shape_init_c-
)gradients_concat_1_grad_mod_concat_1_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_permA
=gradients_transpose_7_grad_invertpermutation_transpose_7_permA
=gradients_transpose_8_grad_invertpermutation_transpose_8_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim)
%gradients_concat_grad_mod_concat_axis
identity

identity_1

identity_2

identity_3

identity_4

identity_5ѕ_
gradients/grad_ys_0Identityplaceholder*
T0*(
_output_shapes
:         аe
gradients/grad_ys_1Identityplaceholder_1*
T0*,
_output_shapes
:         аa
gradients/grad_ys_2Identityplaceholder_2*
T0*(
_output_shapes
:         аa
gradients/grad_ys_3Identityplaceholder_3*
T0*(
_output_shapes
:         аO
gradients/grad_ys_4Identityplaceholder_4*
T0*
_output_shapes
: ђ
!gradients/ExpandDims_2_grad/ShapeShape/gradients_expanddims_2_grad_shape_strided_slice*
T0*
_output_shapes
:Ф
#gradients/ExpandDims_2_grad/ReshapeReshapegradients/grad_ys_1:output:0*gradients/ExpandDims_2_grad/Shape:output:0*
T0*(
_output_shapes
:         аq
gradients/Squeeze_grad/ShapeShape%gradients_squeeze_grad_shape_cudnnrnn*
T0*
_output_shapes
:Ц
gradients/Squeeze_grad/ReshapeReshapegradients/grad_ys_2:output:0%gradients/Squeeze_grad/Shape:output:0*
T0*,
_output_shapes
:         аu
gradients/Squeeze_1_grad/ShapeShape'gradients_squeeze_1_grad_shape_cudnnrnn*
T0*
_output_shapes
:Е
 gradients/Squeeze_1_grad/ReshapeReshapegradients/grad_ys_3:output:0'gradients/Squeeze_1_grad/Shape:output:0*
T0*,
_output_shapes
:         ак
gradients/AddNAddNgradients/grad_ys_0:output:0,gradients/ExpandDims_2_grad/Reshape:output:0*
N*
T0*&
_class
loc:@gradients/grad_ys_0*(
_output_shapes
:         а}
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
:є
3gradients/strided_slice_grad/StridedSliceGrad/beginConst*
_output_shapes
:*
dtype0*
valueB:
         {
1gradients/strided_slice_grad/StridedSliceGrad/endConst*
_output_shapes
:*
dtype0*
valueB: 
5gradients/strided_slice_grad/StridedSliceGrad/stridesConst*
_output_shapes
:*
dtype0*
valueB:џ
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0<gradients/strided_slice_grad/StridedSliceGrad/begin:output:0:gradients/strided_slice_grad/StridedSliceGrad/end:output:0>gradients/strided_slice_grad/StridedSliceGrad/strides:output:0gradients/AddN:sum:0*
Index0*
T0*,
_output_shapes
:         а*
shrink_axis_maskc
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
:Ц
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_11gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1+gradients_strided_slice_grad_shape_cudnnrnn%gradients_squeeze_grad_shape_cudnnrnn'gradients_squeeze_1_grad_shape_cudnnrnn6gradients/strided_slice_grad/StridedSliceGrad:output:0'gradients/Squeeze_grad/Reshape:output:0)gradients/Squeeze_1_grad/Reshape:output:0gradients_zeros_like_cudnnrnn*
T0*c
_output_shapesQ
O:         :         а:         а:ђ╗ќ
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:л
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*+
_output_shapes
:         u
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
:к
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*(
_output_shapes
:         аy
!gradients/ExpandDims_1_grad/ShapeShape(gradients_expanddims_1_grad_shape_init_c*
T0*
_output_shapes
:╩
#gradients/ExpandDims_1_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_c_backprop:0*gradients/ExpandDims_1_grad/Shape:output:0*
T0*(
_output_shapes
:         а^
gradients/concat_1_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :џ
gradients/concat_1_grad/modFloorMod)gradients_concat_1_grad_mod_concat_1_axis%gradients/concat_1_grad/Rank:output:0*
T0*
_output_shapes
: h
gradients/concat_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:аj
gradients/concat_1_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:аj
gradients/concat_1_grad/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:аj
gradients/concat_1_grad/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:аk
gradients/concat_1_grad/Shape_4Const*
_output_shapes
:*
dtype0*
valueB:ђѕk
gradients/concat_1_grad/Shape_5Const*
_output_shapes
:*
dtype0*
valueB:ђѕk
gradients/concat_1_grad/Shape_6Const*
_output_shapes
:*
dtype0*
valueB:ђѕk
gradients/concat_1_grad/Shape_7Const*
_output_shapes
:*
dtype0*
valueB:ђѕj
gradients/concat_1_grad/Shape_8Const*
_output_shapes
:*
dtype0*
valueB:аj
gradients/concat_1_grad/Shape_9Const*
_output_shapes
:*
dtype0*
valueB:аk
 gradients/concat_1_grad/Shape_10Const*
_output_shapes
:*
dtype0*
valueB:аk
 gradients/concat_1_grad/Shape_11Const*
_output_shapes
:*
dtype0*
valueB:аk
 gradients/concat_1_grad/Shape_12Const*
_output_shapes
:*
dtype0*
valueB:аk
 gradients/concat_1_grad/Shape_13Const*
_output_shapes
:*
dtype0*
valueB:аk
 gradients/concat_1_grad/Shape_14Const*
_output_shapes
:*
dtype0*
valueB:аk
 gradients/concat_1_grad/Shape_15Const*
_output_shapes
:*
dtype0*
valueB:аЭ
$gradients/concat_1_grad/ConcatOffsetConcatOffsetgradients/concat_1_grad/mod:z:0&gradients/concat_1_grad/Shape:output:0(gradients/concat_1_grad/Shape_1:output:0(gradients/concat_1_grad/Shape_2:output:0(gradients/concat_1_grad/Shape_3:output:0(gradients/concat_1_grad/Shape_4:output:0(gradients/concat_1_grad/Shape_5:output:0(gradients/concat_1_grad/Shape_6:output:0(gradients/concat_1_grad/Shape_7:output:0(gradients/concat_1_grad/Shape_8:output:0(gradients/concat_1_grad/Shape_9:output:0)gradients/concat_1_grad/Shape_10:output:0)gradients/concat_1_grad/Shape_11:output:0)gradients/concat_1_grad/Shape_12:output:0)gradients/concat_1_grad/Shape_13:output:0)gradients/concat_1_grad/Shape_14:output:0)gradients/concat_1_grad/Shape_15:output:0*
N*t
_output_shapesb
`::::::::::::::::В
gradients/concat_1_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:0&gradients/concat_1_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:а­
gradients/concat_1_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:1(gradients/concat_1_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:а­
gradients/concat_1_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:2(gradients/concat_1_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes	
:а­
gradients/concat_1_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:3(gradients/concat_1_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes	
:аы
gradients/concat_1_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:4(gradients/concat_1_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes

:ђѕы
gradients/concat_1_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:5(gradients/concat_1_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes

:ђѕы
gradients/concat_1_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:6(gradients/concat_1_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes

:ђѕы
gradients/concat_1_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:7(gradients/concat_1_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes

:ђѕ­
gradients/concat_1_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:8(gradients/concat_1_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes	
:а­
gradients/concat_1_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:9(gradients/concat_1_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes	
:аз
 gradients/concat_1_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:10)gradients/concat_1_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes	
:аз
 gradients/concat_1_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:11)gradients/concat_1_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes	
:аз
 gradients/concat_1_grad/Slice_12Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:12)gradients/concat_1_grad/Shape_12:output:0*
Index0*
T0*
_output_shapes	
:аз
 gradients/concat_1_grad/Slice_13Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:13)gradients/concat_1_grad/Shape_13:output:0*
Index0*
T0*
_output_shapes	
:аз
 gradients/concat_1_grad/Slice_14Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:14)gradients/concat_1_grad/Shape_14:output:0*
Index0*
T0*
_output_shapes	
:аз
 gradients/concat_1_grad/Slice_15Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:15)gradients/concat_1_grad/Shape_15:output:0*
Index0*
T0*
_output_shapes	
:аm
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      б
gradients/Reshape_grad/ReshapeReshape&gradients/concat_1_grad/Slice:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes
:	аo
gradients/Reshape_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      е
 gradients/Reshape_1_grad/ReshapeReshape(gradients/concat_1_grad/Slice_1:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0*
_output_shapes
:	аo
gradients/Reshape_2_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      е
 gradients/Reshape_2_grad/ReshapeReshape(gradients/concat_1_grad/Slice_2:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0*
_output_shapes
:	аo
gradients/Reshape_3_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      е
 gradients/Reshape_3_grad/ReshapeReshape(gradients/concat_1_grad/Slice_3:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0*
_output_shapes
:	аo
gradients/Reshape_4_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      Е
 gradients/Reshape_4_grad/ReshapeReshape(gradients/concat_1_grad/Slice_4:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0* 
_output_shapes
:
ааo
gradients/Reshape_5_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      Е
 gradients/Reshape_5_grad/ReshapeReshape(gradients/concat_1_grad/Slice_5:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0* 
_output_shapes
:
ааo
gradients/Reshape_6_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      Е
 gradients/Reshape_6_grad/ReshapeReshape(gradients/concat_1_grad/Slice_6:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0* 
_output_shapes
:
ааo
gradients/Reshape_7_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      Е
 gradients/Reshape_7_grad/ReshapeReshape(gradients/concat_1_grad/Slice_7:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0* 
_output_shapes
:
ааi
gradients/Reshape_8_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:ац
 gradients/Reshape_8_grad/ReshapeReshape(gradients/concat_1_grad/Slice_8:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes	
:аi
gradients/Reshape_9_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:ац
 gradients/Reshape_9_grad/ReshapeReshape(gradients/concat_1_grad/Slice_9:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes	
:аj
gradients/Reshape_10_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:аД
!gradients/Reshape_10_grad/ReshapeReshape)gradients/concat_1_grad/Slice_10:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes	
:аj
gradients/Reshape_11_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:аД
!gradients/Reshape_11_grad/ReshapeReshape)gradients/concat_1_grad/Slice_11:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes	
:аj
gradients/Reshape_12_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:аД
!gradients/Reshape_12_grad/ReshapeReshape)gradients/concat_1_grad/Slice_12:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes	
:аj
gradients/Reshape_13_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:аД
!gradients/Reshape_13_grad/ReshapeReshape)gradients/concat_1_grad/Slice_13:output:0(gradients/Reshape_13_grad/Shape:output:0*
T0*
_output_shapes	
:аj
gradients/Reshape_14_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:аД
!gradients/Reshape_14_grad/ReshapeReshape)gradients/concat_1_grad/Slice_14:output:0(gradients/Reshape_14_grad/Shape:output:0*
T0*
_output_shapes	
:аj
gradients/Reshape_15_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:аД
!gradients/Reshape_15_grad/ReshapeReshape)gradients/concat_1_grad/Slice_15:output:0(gradients/Reshape_15_grad/Shape:output:0*
T0*
_output_shapes	
:аю
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:Х
$gradients/transpose_1_grad/transpose	Transpose'gradients/Reshape_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0*
_output_shapes
:	аю
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:И
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0*
_output_shapes
:	аю
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:И
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0*
_output_shapes
:	аю
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:И
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0*
_output_shapes
:	аю
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:╣
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
ааю
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:╣
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
ааю
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:╣
$gradients/transpose_7_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
ааю
,gradients/transpose_8_grad/InvertPermutationInvertPermutation=gradients_transpose_8_grad_invertpermutation_transpose_8_perm*
_output_shapes
:╣
$gradients/transpose_8_grad/transpose	Transpose)gradients/Reshape_7_grad/Reshape:output:00gradients/transpose_8_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
аау
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0*gradients/Reshape_13_grad/Reshape:output:0*gradients/Reshape_14_grad/Reshape:output:0*gradients/Reshape_15_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
N*
T0*
_output_shapes	
:ђ»
gradients/split_grad/concatConcatV2(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
N*
T0*
_output_shapes
:	ђ	Х
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0(gradients/transpose_7_grad/transpose:y:0(gradients/transpose_8_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
N*
T0* 
_output_shapes
:
ађ	\
gradients/concat_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :њ
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: f
gradients/concat_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:ђ	h
gradients/concat_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:ђ	╩
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0*
N* 
_output_shapes
::м
gradients/concat_grad/SliceSlice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:ђ	о
gradients/concat_grad/Slice_1Slice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:ђ	r
IdentityIdentity&gradients/transpose_grad/transpose:y:0*
T0*+
_output_shapes
:         u

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0*
T0*(
_output_shapes
:         аw

Identity_2Identity,gradients/ExpandDims_1_grad/Reshape:output:0*
T0*(
_output_shapes
:         аf

Identity_3Identity$gradients/split_grad/concat:output:0*
T0*
_output_shapes
:	ђ	i

Identity_4Identity&gradients/split_1_grad/concat:output:0*
T0* 
_output_shapes
:
ађ	d

Identity_5Identity&gradients/concat_grad/Slice_1:output:0*
T0*
_output_shapes	
:ђ	"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*(
_construction_contextkEagerRuntime*ѓ
_input_shapes­
ь:         а:         а:         а:         а: :         а:         а:         а:         а::         :         а:         а:ђ╗::         а:         а: ::::::::: : : : *=
api_implements+)lstm_55c05eb7-11a9-4616-8925-c9051a66537d*
api_preferred_deviceGPU*C
forward_function_name*(__forward_gpu_lstm_with_fallback_1211155*
go_backwards( *

time_major( :. *
(
_output_shapes
:         а:2.
,
_output_shapes
:         а:.*
(
_output_shapes
:         а:.*
(
_output_shapes
:         а:

_output_shapes
: :.*
(
_output_shapes
:         а:2.
,
_output_shapes
:         а:2.
,
_output_shapes
:         а:2.
,
_output_shapes
:         а:	

_output_shapes
::1
-
+
_output_shapes
:         :2.
,
_output_shapes
:         а:2.
,
_output_shapes
:         а:"

_output_shapes

:ђ╗: 

_output_shapes
::.*
(
_output_shapes
:         а:.*
(
_output_shapes
:         а:

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
џ	
к
while_cond_1208679
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice5
1while_while_cond_1208679___redundant_placeholder05
1while_while_cond_1208679___redundant_placeholder15
1while_while_cond_1208679___redundant_placeholder25
1while_while_cond_1208679___redundant_placeholder3
while_identity
`

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :         а:         а: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:         а:.*
(
_output_shapes
:         а:

_output_shapes
: :

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
Ы┬
у
=__inference___backward_gpu_lstm_with_fallback_1208863_1209039
placeholder
placeholder_1
placeholder_2
placeholder_3
placeholder_43
/gradients_expanddims_2_grad_shape_strided_slice)
%gradients_squeeze_grad_shape_cudnnrnn+
'gradients_squeeze_1_grad_shape_cudnnrnn/
+gradients_strided_slice_grad_shape_cudnnrnn!
gradients_zeros_like_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9
5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_15
1gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h,
(gradients_expanddims_1_grad_shape_init_c-
)gradients_concat_1_grad_mod_concat_1_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_permA
=gradients_transpose_7_grad_invertpermutation_transpose_7_permA
=gradients_transpose_8_grad_invertpermutation_transpose_8_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim)
%gradients_concat_grad_mod_concat_axis
identity

identity_1

identity_2

identity_3

identity_4

identity_5ѕ_
gradients/grad_ys_0Identityplaceholder*
T0*(
_output_shapes
:         аe
gradients/grad_ys_1Identityplaceholder_1*
T0*,
_output_shapes
:         аa
gradients/grad_ys_2Identityplaceholder_2*
T0*(
_output_shapes
:         аa
gradients/grad_ys_3Identityplaceholder_3*
T0*(
_output_shapes
:         аO
gradients/grad_ys_4Identityplaceholder_4*
T0*
_output_shapes
: ђ
!gradients/ExpandDims_2_grad/ShapeShape/gradients_expanddims_2_grad_shape_strided_slice*
T0*
_output_shapes
:Ф
#gradients/ExpandDims_2_grad/ReshapeReshapegradients/grad_ys_1:output:0*gradients/ExpandDims_2_grad/Shape:output:0*
T0*(
_output_shapes
:         аq
gradients/Squeeze_grad/ShapeShape%gradients_squeeze_grad_shape_cudnnrnn*
T0*
_output_shapes
:Ц
gradients/Squeeze_grad/ReshapeReshapegradients/grad_ys_2:output:0%gradients/Squeeze_grad/Shape:output:0*
T0*,
_output_shapes
:         аu
gradients/Squeeze_1_grad/ShapeShape'gradients_squeeze_1_grad_shape_cudnnrnn*
T0*
_output_shapes
:Е
 gradients/Squeeze_1_grad/ReshapeReshapegradients/grad_ys_3:output:0'gradients/Squeeze_1_grad/Shape:output:0*
T0*,
_output_shapes
:         ак
gradients/AddNAddNgradients/grad_ys_0:output:0,gradients/ExpandDims_2_grad/Reshape:output:0*
N*
T0*&
_class
loc:@gradients/grad_ys_0*(
_output_shapes
:         а}
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
:є
3gradients/strided_slice_grad/StridedSliceGrad/beginConst*
_output_shapes
:*
dtype0*
valueB:
         {
1gradients/strided_slice_grad/StridedSliceGrad/endConst*
_output_shapes
:*
dtype0*
valueB: 
5gradients/strided_slice_grad/StridedSliceGrad/stridesConst*
_output_shapes
:*
dtype0*
valueB:Б
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0<gradients/strided_slice_grad/StridedSliceGrad/begin:output:0:gradients/strided_slice_grad/StridedSliceGrad/end:output:0>gradients/strided_slice_grad/StridedSliceGrad/strides:output:0gradients/AddN:sum:0*
Index0*
T0*5
_output_shapes#
!:                  а*
shrink_axis_maskc
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
:«
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_11gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1+gradients_strided_slice_grad_shape_cudnnrnn%gradients_squeeze_grad_shape_cudnnrnn'gradients_squeeze_1_grad_shape_cudnnrnn6gradients/strided_slice_grad/StridedSliceGrad:output:0'gradients/Squeeze_grad/Reshape:output:0)gradients/Squeeze_1_grad/Reshape:output:0gradients_zeros_like_cudnnrnn*
T0*l
_output_shapesZ
X:                  :         а:         а:ђ╗ќ
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:┘
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*4
_output_shapes"
 :                  u
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
:к
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*(
_output_shapes
:         аy
!gradients/ExpandDims_1_grad/ShapeShape(gradients_expanddims_1_grad_shape_init_c*
T0*
_output_shapes
:╩
#gradients/ExpandDims_1_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_c_backprop:0*gradients/ExpandDims_1_grad/Shape:output:0*
T0*(
_output_shapes
:         а^
gradients/concat_1_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :џ
gradients/concat_1_grad/modFloorMod)gradients_concat_1_grad_mod_concat_1_axis%gradients/concat_1_grad/Rank:output:0*
T0*
_output_shapes
: h
gradients/concat_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:аj
gradients/concat_1_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:аj
gradients/concat_1_grad/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:аj
gradients/concat_1_grad/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:аk
gradients/concat_1_grad/Shape_4Const*
_output_shapes
:*
dtype0*
valueB:ђѕk
gradients/concat_1_grad/Shape_5Const*
_output_shapes
:*
dtype0*
valueB:ђѕk
gradients/concat_1_grad/Shape_6Const*
_output_shapes
:*
dtype0*
valueB:ђѕk
gradients/concat_1_grad/Shape_7Const*
_output_shapes
:*
dtype0*
valueB:ђѕj
gradients/concat_1_grad/Shape_8Const*
_output_shapes
:*
dtype0*
valueB:аj
gradients/concat_1_grad/Shape_9Const*
_output_shapes
:*
dtype0*
valueB:аk
 gradients/concat_1_grad/Shape_10Const*
_output_shapes
:*
dtype0*
valueB:аk
 gradients/concat_1_grad/Shape_11Const*
_output_shapes
:*
dtype0*
valueB:аk
 gradients/concat_1_grad/Shape_12Const*
_output_shapes
:*
dtype0*
valueB:аk
 gradients/concat_1_grad/Shape_13Const*
_output_shapes
:*
dtype0*
valueB:аk
 gradients/concat_1_grad/Shape_14Const*
_output_shapes
:*
dtype0*
valueB:аk
 gradients/concat_1_grad/Shape_15Const*
_output_shapes
:*
dtype0*
valueB:аЭ
$gradients/concat_1_grad/ConcatOffsetConcatOffsetgradients/concat_1_grad/mod:z:0&gradients/concat_1_grad/Shape:output:0(gradients/concat_1_grad/Shape_1:output:0(gradients/concat_1_grad/Shape_2:output:0(gradients/concat_1_grad/Shape_3:output:0(gradients/concat_1_grad/Shape_4:output:0(gradients/concat_1_grad/Shape_5:output:0(gradients/concat_1_grad/Shape_6:output:0(gradients/concat_1_grad/Shape_7:output:0(gradients/concat_1_grad/Shape_8:output:0(gradients/concat_1_grad/Shape_9:output:0)gradients/concat_1_grad/Shape_10:output:0)gradients/concat_1_grad/Shape_11:output:0)gradients/concat_1_grad/Shape_12:output:0)gradients/concat_1_grad/Shape_13:output:0)gradients/concat_1_grad/Shape_14:output:0)gradients/concat_1_grad/Shape_15:output:0*
N*t
_output_shapesb
`::::::::::::::::В
gradients/concat_1_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:0&gradients/concat_1_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:а­
gradients/concat_1_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:1(gradients/concat_1_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:а­
gradients/concat_1_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:2(gradients/concat_1_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes	
:а­
gradients/concat_1_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:3(gradients/concat_1_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes	
:аы
gradients/concat_1_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:4(gradients/concat_1_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes

:ђѕы
gradients/concat_1_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:5(gradients/concat_1_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes

:ђѕы
gradients/concat_1_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:6(gradients/concat_1_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes

:ђѕы
gradients/concat_1_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:7(gradients/concat_1_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes

:ђѕ­
gradients/concat_1_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:8(gradients/concat_1_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes	
:а­
gradients/concat_1_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:9(gradients/concat_1_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes	
:аз
 gradients/concat_1_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:10)gradients/concat_1_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes	
:аз
 gradients/concat_1_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:11)gradients/concat_1_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes	
:аз
 gradients/concat_1_grad/Slice_12Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:12)gradients/concat_1_grad/Shape_12:output:0*
Index0*
T0*
_output_shapes	
:аз
 gradients/concat_1_grad/Slice_13Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:13)gradients/concat_1_grad/Shape_13:output:0*
Index0*
T0*
_output_shapes	
:аз
 gradients/concat_1_grad/Slice_14Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:14)gradients/concat_1_grad/Shape_14:output:0*
Index0*
T0*
_output_shapes	
:аз
 gradients/concat_1_grad/Slice_15Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:15)gradients/concat_1_grad/Shape_15:output:0*
Index0*
T0*
_output_shapes	
:аm
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      б
gradients/Reshape_grad/ReshapeReshape&gradients/concat_1_grad/Slice:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes
:	аo
gradients/Reshape_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      е
 gradients/Reshape_1_grad/ReshapeReshape(gradients/concat_1_grad/Slice_1:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0*
_output_shapes
:	аo
gradients/Reshape_2_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      е
 gradients/Reshape_2_grad/ReshapeReshape(gradients/concat_1_grad/Slice_2:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0*
_output_shapes
:	аo
gradients/Reshape_3_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      е
 gradients/Reshape_3_grad/ReshapeReshape(gradients/concat_1_grad/Slice_3:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0*
_output_shapes
:	аo
gradients/Reshape_4_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      Е
 gradients/Reshape_4_grad/ReshapeReshape(gradients/concat_1_grad/Slice_4:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0* 
_output_shapes
:
ааo
gradients/Reshape_5_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      Е
 gradients/Reshape_5_grad/ReshapeReshape(gradients/concat_1_grad/Slice_5:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0* 
_output_shapes
:
ааo
gradients/Reshape_6_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      Е
 gradients/Reshape_6_grad/ReshapeReshape(gradients/concat_1_grad/Slice_6:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0* 
_output_shapes
:
ааo
gradients/Reshape_7_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      Е
 gradients/Reshape_7_grad/ReshapeReshape(gradients/concat_1_grad/Slice_7:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0* 
_output_shapes
:
ааi
gradients/Reshape_8_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:ац
 gradients/Reshape_8_grad/ReshapeReshape(gradients/concat_1_grad/Slice_8:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes	
:аi
gradients/Reshape_9_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:ац
 gradients/Reshape_9_grad/ReshapeReshape(gradients/concat_1_grad/Slice_9:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes	
:аj
gradients/Reshape_10_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:аД
!gradients/Reshape_10_grad/ReshapeReshape)gradients/concat_1_grad/Slice_10:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes	
:аj
gradients/Reshape_11_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:аД
!gradients/Reshape_11_grad/ReshapeReshape)gradients/concat_1_grad/Slice_11:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes	
:аj
gradients/Reshape_12_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:аД
!gradients/Reshape_12_grad/ReshapeReshape)gradients/concat_1_grad/Slice_12:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes	
:аj
gradients/Reshape_13_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:аД
!gradients/Reshape_13_grad/ReshapeReshape)gradients/concat_1_grad/Slice_13:output:0(gradients/Reshape_13_grad/Shape:output:0*
T0*
_output_shapes	
:аj
gradients/Reshape_14_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:аД
!gradients/Reshape_14_grad/ReshapeReshape)gradients/concat_1_grad/Slice_14:output:0(gradients/Reshape_14_grad/Shape:output:0*
T0*
_output_shapes	
:аj
gradients/Reshape_15_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:аД
!gradients/Reshape_15_grad/ReshapeReshape)gradients/concat_1_grad/Slice_15:output:0(gradients/Reshape_15_grad/Shape:output:0*
T0*
_output_shapes	
:аю
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:Х
$gradients/transpose_1_grad/transpose	Transpose'gradients/Reshape_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0*
_output_shapes
:	аю
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:И
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0*
_output_shapes
:	аю
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:И
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0*
_output_shapes
:	аю
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:И
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0*
_output_shapes
:	аю
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:╣
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
ааю
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:╣
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
ааю
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:╣
$gradients/transpose_7_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
ааю
,gradients/transpose_8_grad/InvertPermutationInvertPermutation=gradients_transpose_8_grad_invertpermutation_transpose_8_perm*
_output_shapes
:╣
$gradients/transpose_8_grad/transpose	Transpose)gradients/Reshape_7_grad/Reshape:output:00gradients/transpose_8_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
аау
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0*gradients/Reshape_13_grad/Reshape:output:0*gradients/Reshape_14_grad/Reshape:output:0*gradients/Reshape_15_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
N*
T0*
_output_shapes	
:ђ»
gradients/split_grad/concatConcatV2(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
N*
T0*
_output_shapes
:	ђ	Х
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0(gradients/transpose_7_grad/transpose:y:0(gradients/transpose_8_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
N*
T0* 
_output_shapes
:
ађ	\
gradients/concat_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :њ
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: f
gradients/concat_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:ђ	h
gradients/concat_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:ђ	╩
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0*
N* 
_output_shapes
::м
gradients/concat_grad/SliceSlice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:ђ	о
gradients/concat_grad/Slice_1Slice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:ђ	{
IdentityIdentity&gradients/transpose_grad/transpose:y:0*
T0*4
_output_shapes"
 :                  u

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0*
T0*(
_output_shapes
:         аw

Identity_2Identity,gradients/ExpandDims_1_grad/Reshape:output:0*
T0*(
_output_shapes
:         аf

Identity_3Identity$gradients/split_grad/concat:output:0*
T0*
_output_shapes
:	ђ	i

Identity_4Identity&gradients/split_1_grad/concat:output:0*
T0* 
_output_shapes
:
ађ	d

Identity_5Identity&gradients/concat_grad/Slice_1:output:0*
T0*
_output_shapes	
:ђ	"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*(
_construction_contextkEagerRuntime*ћ
_input_shapesѓ
 :         а:         а:         а:         а: :         а:         а:         а:                  а::                  :         а:         а:ђ╗::         а:         а: ::::::::: : : : *=
api_implements+)lstm_8b39bb2f-e1e5-4618-9947-ffe4540afe4b*
api_preferred_deviceGPU*C
forward_function_name*(__forward_gpu_lstm_with_fallback_1209038*
go_backwards( *

time_major( :. *
(
_output_shapes
:         а:2.
,
_output_shapes
:         а:.*
(
_output_shapes
:         а:.*
(
_output_shapes
:         а:

_output_shapes
: :.*
(
_output_shapes
:         а:2.
,
_output_shapes
:         а:2.
,
_output_shapes
:         а:;7
5
_output_shapes#
!:                  а:	

_output_shapes
:::
6
4
_output_shapes"
 :                  :2.
,
_output_shapes
:         а:2.
,
_output_shapes
:         а:"

_output_shapes

:ђ╗: 

_output_shapes
::.*
(
_output_shapes
:         а:.*
(
_output_shapes
:         а:

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
џ	
к
while_cond_1212640
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice5
1while_while_cond_1212640___redundant_placeholder05
1while_while_cond_1212640___redundant_placeholder15
1while_while_cond_1212640___redundant_placeholder25
1while_while_cond_1212640___redundant_placeholder3
while_identity
`

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :         а:         а: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:         а:.*
(
_output_shapes
:         а:

_output_shapes
: :

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
ф┬
у
=__inference___backward_gpu_lstm_with_fallback_1210533_1210709
placeholder
placeholder_1
placeholder_2
placeholder_3
placeholder_43
/gradients_expanddims_2_grad_shape_strided_slice)
%gradients_squeeze_grad_shape_cudnnrnn+
'gradients_squeeze_1_grad_shape_cudnnrnn/
+gradients_strided_slice_grad_shape_cudnnrnn!
gradients_zeros_like_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9
5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_15
1gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h,
(gradients_expanddims_1_grad_shape_init_c-
)gradients_concat_1_grad_mod_concat_1_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_permA
=gradients_transpose_7_grad_invertpermutation_transpose_7_permA
=gradients_transpose_8_grad_invertpermutation_transpose_8_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim)
%gradients_concat_grad_mod_concat_axis
identity

identity_1

identity_2

identity_3

identity_4

identity_5ѕ_
gradients/grad_ys_0Identityplaceholder*
T0*(
_output_shapes
:         аe
gradients/grad_ys_1Identityplaceholder_1*
T0*,
_output_shapes
:         аa
gradients/grad_ys_2Identityplaceholder_2*
T0*(
_output_shapes
:         аa
gradients/grad_ys_3Identityplaceholder_3*
T0*(
_output_shapes
:         аO
gradients/grad_ys_4Identityplaceholder_4*
T0*
_output_shapes
: ђ
!gradients/ExpandDims_2_grad/ShapeShape/gradients_expanddims_2_grad_shape_strided_slice*
T0*
_output_shapes
:Ф
#gradients/ExpandDims_2_grad/ReshapeReshapegradients/grad_ys_1:output:0*gradients/ExpandDims_2_grad/Shape:output:0*
T0*(
_output_shapes
:         аq
gradients/Squeeze_grad/ShapeShape%gradients_squeeze_grad_shape_cudnnrnn*
T0*
_output_shapes
:Ц
gradients/Squeeze_grad/ReshapeReshapegradients/grad_ys_2:output:0%gradients/Squeeze_grad/Shape:output:0*
T0*,
_output_shapes
:         аu
gradients/Squeeze_1_grad/ShapeShape'gradients_squeeze_1_grad_shape_cudnnrnn*
T0*
_output_shapes
:Е
 gradients/Squeeze_1_grad/ReshapeReshapegradients/grad_ys_3:output:0'gradients/Squeeze_1_grad/Shape:output:0*
T0*,
_output_shapes
:         ак
gradients/AddNAddNgradients/grad_ys_0:output:0,gradients/ExpandDims_2_grad/Reshape:output:0*
N*
T0*&
_class
loc:@gradients/grad_ys_0*(
_output_shapes
:         а}
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
:є
3gradients/strided_slice_grad/StridedSliceGrad/beginConst*
_output_shapes
:*
dtype0*
valueB:
         {
1gradients/strided_slice_grad/StridedSliceGrad/endConst*
_output_shapes
:*
dtype0*
valueB: 
5gradients/strided_slice_grad/StridedSliceGrad/stridesConst*
_output_shapes
:*
dtype0*
valueB:џ
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0<gradients/strided_slice_grad/StridedSliceGrad/begin:output:0:gradients/strided_slice_grad/StridedSliceGrad/end:output:0>gradients/strided_slice_grad/StridedSliceGrad/strides:output:0gradients/AddN:sum:0*
Index0*
T0*,
_output_shapes
:         а*
shrink_axis_maskc
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
:Ц
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_11gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1+gradients_strided_slice_grad_shape_cudnnrnn%gradients_squeeze_grad_shape_cudnnrnn'gradients_squeeze_1_grad_shape_cudnnrnn6gradients/strided_slice_grad/StridedSliceGrad:output:0'gradients/Squeeze_grad/Reshape:output:0)gradients/Squeeze_1_grad/Reshape:output:0gradients_zeros_like_cudnnrnn*
T0*c
_output_shapesQ
O:         :         а:         а:ђ╗ќ
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:л
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*+
_output_shapes
:         u
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
:к
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*(
_output_shapes
:         аy
!gradients/ExpandDims_1_grad/ShapeShape(gradients_expanddims_1_grad_shape_init_c*
T0*
_output_shapes
:╩
#gradients/ExpandDims_1_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_c_backprop:0*gradients/ExpandDims_1_grad/Shape:output:0*
T0*(
_output_shapes
:         а^
gradients/concat_1_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :џ
gradients/concat_1_grad/modFloorMod)gradients_concat_1_grad_mod_concat_1_axis%gradients/concat_1_grad/Rank:output:0*
T0*
_output_shapes
: h
gradients/concat_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:аj
gradients/concat_1_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:аj
gradients/concat_1_grad/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:аj
gradients/concat_1_grad/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:аk
gradients/concat_1_grad/Shape_4Const*
_output_shapes
:*
dtype0*
valueB:ђѕk
gradients/concat_1_grad/Shape_5Const*
_output_shapes
:*
dtype0*
valueB:ђѕk
gradients/concat_1_grad/Shape_6Const*
_output_shapes
:*
dtype0*
valueB:ђѕk
gradients/concat_1_grad/Shape_7Const*
_output_shapes
:*
dtype0*
valueB:ђѕj
gradients/concat_1_grad/Shape_8Const*
_output_shapes
:*
dtype0*
valueB:аj
gradients/concat_1_grad/Shape_9Const*
_output_shapes
:*
dtype0*
valueB:аk
 gradients/concat_1_grad/Shape_10Const*
_output_shapes
:*
dtype0*
valueB:аk
 gradients/concat_1_grad/Shape_11Const*
_output_shapes
:*
dtype0*
valueB:аk
 gradients/concat_1_grad/Shape_12Const*
_output_shapes
:*
dtype0*
valueB:аk
 gradients/concat_1_grad/Shape_13Const*
_output_shapes
:*
dtype0*
valueB:аk
 gradients/concat_1_grad/Shape_14Const*
_output_shapes
:*
dtype0*
valueB:аk
 gradients/concat_1_grad/Shape_15Const*
_output_shapes
:*
dtype0*
valueB:аЭ
$gradients/concat_1_grad/ConcatOffsetConcatOffsetgradients/concat_1_grad/mod:z:0&gradients/concat_1_grad/Shape:output:0(gradients/concat_1_grad/Shape_1:output:0(gradients/concat_1_grad/Shape_2:output:0(gradients/concat_1_grad/Shape_3:output:0(gradients/concat_1_grad/Shape_4:output:0(gradients/concat_1_grad/Shape_5:output:0(gradients/concat_1_grad/Shape_6:output:0(gradients/concat_1_grad/Shape_7:output:0(gradients/concat_1_grad/Shape_8:output:0(gradients/concat_1_grad/Shape_9:output:0)gradients/concat_1_grad/Shape_10:output:0)gradients/concat_1_grad/Shape_11:output:0)gradients/concat_1_grad/Shape_12:output:0)gradients/concat_1_grad/Shape_13:output:0)gradients/concat_1_grad/Shape_14:output:0)gradients/concat_1_grad/Shape_15:output:0*
N*t
_output_shapesb
`::::::::::::::::В
gradients/concat_1_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:0&gradients/concat_1_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:а­
gradients/concat_1_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:1(gradients/concat_1_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:а­
gradients/concat_1_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:2(gradients/concat_1_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes	
:а­
gradients/concat_1_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:3(gradients/concat_1_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes	
:аы
gradients/concat_1_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:4(gradients/concat_1_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes

:ђѕы
gradients/concat_1_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:5(gradients/concat_1_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes

:ђѕы
gradients/concat_1_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:6(gradients/concat_1_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes

:ђѕы
gradients/concat_1_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:7(gradients/concat_1_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes

:ђѕ­
gradients/concat_1_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:8(gradients/concat_1_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes	
:а­
gradients/concat_1_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:9(gradients/concat_1_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes	
:аз
 gradients/concat_1_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:10)gradients/concat_1_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes	
:аз
 gradients/concat_1_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:11)gradients/concat_1_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes	
:аз
 gradients/concat_1_grad/Slice_12Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:12)gradients/concat_1_grad/Shape_12:output:0*
Index0*
T0*
_output_shapes	
:аз
 gradients/concat_1_grad/Slice_13Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:13)gradients/concat_1_grad/Shape_13:output:0*
Index0*
T0*
_output_shapes	
:аз
 gradients/concat_1_grad/Slice_14Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:14)gradients/concat_1_grad/Shape_14:output:0*
Index0*
T0*
_output_shapes	
:аз
 gradients/concat_1_grad/Slice_15Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:15)gradients/concat_1_grad/Shape_15:output:0*
Index0*
T0*
_output_shapes	
:аm
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      б
gradients/Reshape_grad/ReshapeReshape&gradients/concat_1_grad/Slice:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes
:	аo
gradients/Reshape_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      е
 gradients/Reshape_1_grad/ReshapeReshape(gradients/concat_1_grad/Slice_1:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0*
_output_shapes
:	аo
gradients/Reshape_2_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      е
 gradients/Reshape_2_grad/ReshapeReshape(gradients/concat_1_grad/Slice_2:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0*
_output_shapes
:	аo
gradients/Reshape_3_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      е
 gradients/Reshape_3_grad/ReshapeReshape(gradients/concat_1_grad/Slice_3:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0*
_output_shapes
:	аo
gradients/Reshape_4_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      Е
 gradients/Reshape_4_grad/ReshapeReshape(gradients/concat_1_grad/Slice_4:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0* 
_output_shapes
:
ааo
gradients/Reshape_5_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      Е
 gradients/Reshape_5_grad/ReshapeReshape(gradients/concat_1_grad/Slice_5:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0* 
_output_shapes
:
ааo
gradients/Reshape_6_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      Е
 gradients/Reshape_6_grad/ReshapeReshape(gradients/concat_1_grad/Slice_6:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0* 
_output_shapes
:
ааo
gradients/Reshape_7_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      Е
 gradients/Reshape_7_grad/ReshapeReshape(gradients/concat_1_grad/Slice_7:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0* 
_output_shapes
:
ааi
gradients/Reshape_8_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:ац
 gradients/Reshape_8_grad/ReshapeReshape(gradients/concat_1_grad/Slice_8:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes	
:аi
gradients/Reshape_9_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:ац
 gradients/Reshape_9_grad/ReshapeReshape(gradients/concat_1_grad/Slice_9:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes	
:аj
gradients/Reshape_10_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:аД
!gradients/Reshape_10_grad/ReshapeReshape)gradients/concat_1_grad/Slice_10:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes	
:аj
gradients/Reshape_11_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:аД
!gradients/Reshape_11_grad/ReshapeReshape)gradients/concat_1_grad/Slice_11:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes	
:аj
gradients/Reshape_12_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:аД
!gradients/Reshape_12_grad/ReshapeReshape)gradients/concat_1_grad/Slice_12:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes	
:аj
gradients/Reshape_13_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:аД
!gradients/Reshape_13_grad/ReshapeReshape)gradients/concat_1_grad/Slice_13:output:0(gradients/Reshape_13_grad/Shape:output:0*
T0*
_output_shapes	
:аj
gradients/Reshape_14_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:аД
!gradients/Reshape_14_grad/ReshapeReshape)gradients/concat_1_grad/Slice_14:output:0(gradients/Reshape_14_grad/Shape:output:0*
T0*
_output_shapes	
:аj
gradients/Reshape_15_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:аД
!gradients/Reshape_15_grad/ReshapeReshape)gradients/concat_1_grad/Slice_15:output:0(gradients/Reshape_15_grad/Shape:output:0*
T0*
_output_shapes	
:аю
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:Х
$gradients/transpose_1_grad/transpose	Transpose'gradients/Reshape_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0*
_output_shapes
:	аю
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:И
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0*
_output_shapes
:	аю
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:И
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0*
_output_shapes
:	аю
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:И
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0*
_output_shapes
:	аю
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:╣
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
ааю
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:╣
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
ааю
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:╣
$gradients/transpose_7_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
ааю
,gradients/transpose_8_grad/InvertPermutationInvertPermutation=gradients_transpose_8_grad_invertpermutation_transpose_8_perm*
_output_shapes
:╣
$gradients/transpose_8_grad/transpose	Transpose)gradients/Reshape_7_grad/Reshape:output:00gradients/transpose_8_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
аау
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0*gradients/Reshape_13_grad/Reshape:output:0*gradients/Reshape_14_grad/Reshape:output:0*gradients/Reshape_15_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
N*
T0*
_output_shapes	
:ђ»
gradients/split_grad/concatConcatV2(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
N*
T0*
_output_shapes
:	ђ	Х
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0(gradients/transpose_7_grad/transpose:y:0(gradients/transpose_8_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
N*
T0* 
_output_shapes
:
ађ	\
gradients/concat_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :њ
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: f
gradients/concat_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:ђ	h
gradients/concat_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:ђ	╩
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0*
N* 
_output_shapes
::м
gradients/concat_grad/SliceSlice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:ђ	о
gradients/concat_grad/Slice_1Slice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:ђ	r
IdentityIdentity&gradients/transpose_grad/transpose:y:0*
T0*+
_output_shapes
:         u

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0*
T0*(
_output_shapes
:         аw

Identity_2Identity,gradients/ExpandDims_1_grad/Reshape:output:0*
T0*(
_output_shapes
:         аf

Identity_3Identity$gradients/split_grad/concat:output:0*
T0*
_output_shapes
:	ђ	i

Identity_4Identity&gradients/split_1_grad/concat:output:0*
T0* 
_output_shapes
:
ађ	d

Identity_5Identity&gradients/concat_grad/Slice_1:output:0*
T0*
_output_shapes	
:ђ	"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*(
_construction_contextkEagerRuntime*ѓ
_input_shapes­
ь:         а:         а:         а:         а: :         а:         а:         а:         а::         :         а:         а:ђ╗::         а:         а: ::::::::: : : : *=
api_implements+)lstm_eb22870e-927f-49a2-b687-85a1e58290d1*
api_preferred_deviceGPU*C
forward_function_name*(__forward_gpu_lstm_with_fallback_1210708*
go_backwards( *

time_major( :. *
(
_output_shapes
:         а:2.
,
_output_shapes
:         а:.*
(
_output_shapes
:         а:.*
(
_output_shapes
:         а:

_output_shapes
: :.*
(
_output_shapes
:         а:2.
,
_output_shapes
:         а:2.
,
_output_shapes
:         а:2.
,
_output_shapes
:         а:	

_output_shapes
::1
-
+
_output_shapes
:         :2.
,
_output_shapes
:         а:2.
,
_output_shapes
:         а:"

_output_shapes

:ђ╗: 

_output_shapes
::.*
(
_output_shapes
:         а:.*
(
_output_shapes
:         а:

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
эM
б
(__forward_gpu_lstm_with_fallback_1211632

inputs
init_h_0
init_c_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4
strided_slice
cudnnrnn

cudnnrnn_0

cudnnrnn_1

cudnnrnn_2
	transpose

expanddims
expanddims_1
concat_1
transpose_perm

init_h

init_c
concat_1_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
transpose_7_perm
transpose_8_perm
split_2_split_dim
split_split_dim
split_1_split_dim
concat_axisѕc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          B
transpose_0	Transposeinputstranspose/perm:output:0*
T0P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : r

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T0*,
_output_shapes
:         аR
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : v
ExpandDims_1
ExpandDimsinit_c_0ExpandDims_1/dim:output:0*
T0*,
_output_shapes
:         аQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ї
splitSplitsplit/split_dim:output:0kernel*
T0*@
_output_shapes.
,:	а:	а:	а:	а*
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ъ
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*D
_output_shapes2
0:
аа:
аа:
аа:
аа*
	num_splite
zeros_like/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB:ђ	U
zeros_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    x

zeros_likeFill#zeros_like/shape_as_tensor:output:0zeros_like/Const:output:0*
T0*
_output_shapes	
:ђ	M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : r
concatConcatV2zeros_like:output:0biasconcat/axis:output:0*
N*
T0*
_output_shapes	
:ђS
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : Ц
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*L
_output_shapes:
8:а:а:а:а:а:а:а:а*
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
         a
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       m
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes
:	аY
ReshapeReshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:аa
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       m
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes
:	а[
	Reshape_1Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:аa
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       m
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes
:	а[
	Reshape_2Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:аa
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       m
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0*
_output_shapes
:	а[
	Reshape_3Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes	
:аa
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       p
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0* 
_output_shapes
:
аа\
	Reshape_4Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes

:ђѕa
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       p
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0* 
_output_shapes
:
аа\
	Reshape_5Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes

:ђѕa
transpose_7/permConst*
_output_shapes
:*
dtype0*
valueB"       p
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0* 
_output_shapes
:
аа\
	Reshape_6Reshapetranspose_7:y:0Const:output:0*
T0*
_output_shapes

:ђѕa
transpose_8/permConst*
_output_shapes
:*
dtype0*
valueB"       p
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0* 
_output_shapes
:
аа\
	Reshape_7Reshapetranspose_8:y:0Const:output:0*
T0*
_output_shapes

:ђѕ\
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes	
:а\
	Reshape_9Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes	
:а]

Reshape_10Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes	
:а]

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes	
:а]

Reshape_12Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes	
:а]

Reshape_13Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes	
:а]

Reshape_14Reshapesplit_2:output:6Const:output:0*
T0*
_output_shapes	
:а]

Reshape_15Reshapesplit_2:output:7Const:output:0*
T0*
_output_shapes	
:аO
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ё

concat_1_0ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T0Л
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1_0:output:0*
T0*`
_output_shapesN
L:         а:         а:         а:f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
         _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Й
strided_slice_0StridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
shrink_axis_maskq
SqueezeSqueezeCudnnRNN:output_h:0*
T0*(
_output_shapes
:         а*
squeeze_dims
 s
	Squeeze_1SqueezeCudnnRNN:output_c:0*
T0*(
_output_shapes
:         а*
squeeze_dims
 R
ExpandDims_2/dimConst*
_output_shapes
: *
dtype0*
value	B :є
ExpandDims_2
ExpandDimsstrided_slice_0:output:0ExpandDims_2/dim:output:0*
T0*,
_output_shapes
:         а[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @a
IdentityIdentitystrided_slice_0:output:0*
T0*(
_output_shapes
:         аd

Identity_1IdentityExpandDims_2:output:0*
T0*,
_output_shapes
:         а[

Identity_2IdentitySqueeze:output:0*
T0*(
_output_shapes
:         а]

Identity_3IdentitySqueeze_1:output:0*
T0*(
_output_shapes
:         аI

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: "
concat_1concat_1_0:output:0"'
concat_1_axisconcat_1/axis:output:0"#
concat_axisconcat/axis:output:0"
cudnnrnnCudnnRNN:output_h:0"!

cudnnrnn_0CudnnRNN:output_c:0"

cudnnrnn_1CudnnRNN:output:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"!

expanddimsExpandDims:output:0"%
expanddims_1ExpandDims_1:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
init_cinit_c_0"
init_hinit_h_0"/
split_1_split_dimsplit_1/split_dim:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"+
split_split_dimsplit/split_dim:output:0")
strided_slicestrided_slice_0:output:0"
	transposetranspose_0:y:0"-
transpose_1_permtranspose_1/perm:output:0"-
transpose_2_permtranspose_2/perm:output:0"-
transpose_3_permtranspose_3/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"-
transpose_5_permtranspose_5/perm:output:0"-
transpose_6_permtranspose_6/perm:output:0"-
transpose_7_permtranspose_7/perm:output:0"-
transpose_8_permtranspose_8/perm:output:0")
transpose_permtranspose/perm:output:0*(
_construction_contextkEagerRuntime*p
_input_shapes_
]:         :         а:         а:	ђ	:
ађ	:ђ	*=
api_implements+)lstm_cbe182d0-608f-49e0-986e-6c0a38bbebc2*
api_preferred_deviceGPU*Y
backward_function_name?=__inference___backward_gpu_lstm_with_fallback_1211457_1211633*
go_backwards( *

time_major( :S O
+
_output_shapes
:         
 
_user_specified_nameinputs:PL
(
_output_shapes
:         а
 
_user_specified_nameinit_h:PL
(
_output_shapes
:         а
 
_user_specified_nameinit_c:GC

_output_shapes
:	ђ	
 
_user_specified_namekernel:RN
 
_output_shapes
:
ађ	
*
_user_specified_namerecurrent_kernel:A=

_output_shapes	
:ђ	

_user_specified_namebias
џ	
к
while_cond_1211273
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice5
1while_while_cond_1211273___redundant_placeholder05
1while_while_cond_1211273___redundant_placeholder15
1while_while_cond_1211273___redundant_placeholder25
1while_while_cond_1211273___redundant_placeholder3
while_identity
`

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :         а:         а: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:         а:.*
(
_output_shapes
:         а:

_output_shapes
: :

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
џ	
к
while_cond_1209638
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice5
1while_while_cond_1209638___redundant_placeholder05
1while_while_cond_1209638___redundant_placeholder15
1while_while_cond_1209638___redundant_placeholder25
1while_while_cond_1209638___redundant_placeholder3
while_identity
`

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :         а:         а: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:         а:.*
(
_output_shapes
:         а:

_output_shapes
: :

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
Б
┐
D__inference_lstm_13_layer_call_and_return_conditional_losses_1209487

inputs/
read_readvariableop_resource:	ђ	2
read_1_readvariableop_resource:
ађ	-
read_2_readvariableop_resource:	ђ	

identity_3ѕбRead/ReadVariableOpбRead_1/ReadVariableOpбRead_2/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Л
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :аs
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:         аS
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :аw
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    s
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:         аq
Read/ReadVariableOpReadVariableOpread_readvariableop_resource*
_output_shapes
:	ђ	*
dtype0[
IdentityIdentityRead/ReadVariableOp:value:0*
T0*
_output_shapes
:	ђ	v
Read_1/ReadVariableOpReadVariableOpread_1_readvariableop_resource* 
_output_shapes
:
ађ	*
dtype0`

Identity_1IdentityRead_1/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ађ	q
Read_2/ReadVariableOpReadVariableOpread_2_readvariableop_resource*
_output_shapes	
:ђ	*
dtype0[

Identity_2IdentityRead_2/ReadVariableOp:value:0*
T0*
_output_shapes	
:ђ	и
PartitionedCallPartitionedCallinputszeros:output:0zeros_1:output:0Identity:output:0Identity_1:output:0Identity_2:output:0*
Tin

2*
Tout	
2*
_collective_manager_ids
 *j
_output_shapesX
V:         а:         а:         а:         а: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ **
f%R#
!__inference_standard_lstm_1209212j

Identity_3IdentityPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         аї
NoOpNoOp^Read/ReadVariableOp^Read_1/ReadVariableOp^Read_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         : : : 2*
Read/ReadVariableOpRead/ReadVariableOp2.
Read_1/ReadVariableOpRead_1/ReadVariableOp2.
Read_2/ReadVariableOpRead_2/ReadVariableOp:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
┴B
═
*__inference_gpu_lstm_with_fallback_1209821

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4ѕc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:         P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : p

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0*,
_output_shapes
:         аR
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : t
ExpandDims_1
ExpandDimsinit_cExpandDims_1/dim:output:0*
T0*,
_output_shapes
:         аQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ї
splitSplitsplit/split_dim:output:0kernel*
T0*@
_output_shapes.
,:	а:	а:	а:	а*
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ъ
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*D
_output_shapes2
0:
аа:
аа:
аа:
аа*
	num_splite
zeros_like/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB:ђ	U
zeros_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    x

zeros_likeFill#zeros_like/shape_as_tensor:output:0zeros_like/Const:output:0*
T0*
_output_shapes	
:ђ	M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : r
concatConcatV2zeros_like:output:0biasconcat/axis:output:0*
N*
T0*
_output_shapes	
:ђS
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : Ц
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*L
_output_shapes:
8:а:а:а:а:а:а:а:а*
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
         a
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       m
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes
:	аY
ReshapeReshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:аa
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       m
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes
:	а[
	Reshape_1Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:аa
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       m
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes
:	а[
	Reshape_2Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:аa
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       m
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0*
_output_shapes
:	а[
	Reshape_3Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes	
:аa
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       p
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0* 
_output_shapes
:
аа\
	Reshape_4Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes

:ђѕa
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       p
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0* 
_output_shapes
:
аа\
	Reshape_5Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes

:ђѕa
transpose_7/permConst*
_output_shapes
:*
dtype0*
valueB"       p
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0* 
_output_shapes
:
аа\
	Reshape_6Reshapetranspose_7:y:0Const:output:0*
T0*
_output_shapes

:ђѕa
transpose_8/permConst*
_output_shapes
:*
dtype0*
valueB"       p
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0* 
_output_shapes
:
аа\
	Reshape_7Reshapetranspose_8:y:0Const:output:0*
T0*
_output_shapes

:ђѕ\
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes	
:а\
	Reshape_9Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes	
:а]

Reshape_10Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes	
:а]

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes	
:а]

Reshape_12Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes	
:а]

Reshape_13Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes	
:а]

Reshape_14Reshapesplit_2:output:6Const:output:0*
T0*
_output_shapes	
:а]

Reshape_15Reshapesplit_2:output:7Const:output:0*
T0*
_output_shapes	
:аO
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : а
concat_1ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T0*
_output_shapes

:ђ╗═
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1:output:0*
T0*`
_output_shapesN
L:         а:         а:         а:f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
         _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Т
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         а*
shrink_axis_maskq
SqueezeSqueezeCudnnRNN:output_h:0*
T0*(
_output_shapes
:         а*
squeeze_dims
 s
	Squeeze_1SqueezeCudnnRNN:output_c:0*
T0*(
_output_shapes
:         а*
squeeze_dims
 R
ExpandDims_2/dimConst*
_output_shapes
: *
dtype0*
value	B :ё
ExpandDims_2
ExpandDimsstrided_slice:output:0ExpandDims_2/dim:output:0*
T0*,
_output_shapes
:         а[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @_
IdentityIdentitystrided_slice:output:0*
T0*(
_output_shapes
:         аd

Identity_1IdentityExpandDims_2:output:0*
T0*,
_output_shapes
:         а[

Identity_2IdentitySqueeze:output:0*
T0*(
_output_shapes
:         а]

Identity_3IdentitySqueeze_1:output:0*
T0*(
_output_shapes
:         аI

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*p
_input_shapes_
]:         :         а:         а:	ђ	:
ађ	:ђ	*=
api_implements+)lstm_6b3f42d6-117f-4a93-968a-eb0630b50933*
api_preferred_deviceGPU*
go_backwards( *

time_major( :S O
+
_output_shapes
:         
 
_user_specified_nameinputs:PL
(
_output_shapes
:         а
 
_user_specified_nameinit_h:PL
(
_output_shapes
:         а
 
_user_specified_nameinit_c:GC

_output_shapes
:	ђ	
 
_user_specified_namekernel:RN
 
_output_shapes
:
ађ	
*
_user_specified_namerecurrent_kernel:A=

_output_shapes	
:ђ	

_user_specified_namebias
эM
б
(__forward_gpu_lstm_with_fallback_1213861

inputs
init_h_0
init_c_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4
strided_slice
cudnnrnn

cudnnrnn_0

cudnnrnn_1

cudnnrnn_2
	transpose

expanddims
expanddims_1
concat_1
transpose_perm

init_h

init_c
concat_1_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
transpose_7_perm
transpose_8_perm
split_2_split_dim
split_split_dim
split_1_split_dim
concat_axisѕc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          B
transpose_0	Transposeinputstranspose/perm:output:0*
T0P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : r

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T0*,
_output_shapes
:         аR
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : v
ExpandDims_1
ExpandDimsinit_c_0ExpandDims_1/dim:output:0*
T0*,
_output_shapes
:         аQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ї
splitSplitsplit/split_dim:output:0kernel*
T0*@
_output_shapes.
,:	а:	а:	а:	а*
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ъ
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*D
_output_shapes2
0:
аа:
аа:
аа:
аа*
	num_splite
zeros_like/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB:ђ	U
zeros_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    x

zeros_likeFill#zeros_like/shape_as_tensor:output:0zeros_like/Const:output:0*
T0*
_output_shapes	
:ђ	M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : r
concatConcatV2zeros_like:output:0biasconcat/axis:output:0*
N*
T0*
_output_shapes	
:ђS
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : Ц
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*L
_output_shapes:
8:а:а:а:а:а:а:а:а*
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
         a
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       m
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes
:	аY
ReshapeReshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:аa
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       m
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes
:	а[
	Reshape_1Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:аa
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       m
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes
:	а[
	Reshape_2Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:аa
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       m
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0*
_output_shapes
:	а[
	Reshape_3Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes	
:аa
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       p
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0* 
_output_shapes
:
аа\
	Reshape_4Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes

:ђѕa
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       p
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0* 
_output_shapes
:
аа\
	Reshape_5Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes

:ђѕa
transpose_7/permConst*
_output_shapes
:*
dtype0*
valueB"       p
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0* 
_output_shapes
:
аа\
	Reshape_6Reshapetranspose_7:y:0Const:output:0*
T0*
_output_shapes

:ђѕa
transpose_8/permConst*
_output_shapes
:*
dtype0*
valueB"       p
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0* 
_output_shapes
:
аа\
	Reshape_7Reshapetranspose_8:y:0Const:output:0*
T0*
_output_shapes

:ђѕ\
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes	
:а\
	Reshape_9Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes	
:а]

Reshape_10Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes	
:а]

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes	
:а]

Reshape_12Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes	
:а]

Reshape_13Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes	
:а]

Reshape_14Reshapesplit_2:output:6Const:output:0*
T0*
_output_shapes	
:а]

Reshape_15Reshapesplit_2:output:7Const:output:0*
T0*
_output_shapes	
:аO
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ё

concat_1_0ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T0Л
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1_0:output:0*
T0*`
_output_shapesN
L:         а:         а:         а:f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
         _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Й
strided_slice_0StridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
shrink_axis_maskq
SqueezeSqueezeCudnnRNN:output_h:0*
T0*(
_output_shapes
:         а*
squeeze_dims
 s
	Squeeze_1SqueezeCudnnRNN:output_c:0*
T0*(
_output_shapes
:         а*
squeeze_dims
 R
ExpandDims_2/dimConst*
_output_shapes
: *
dtype0*
value	B :є
ExpandDims_2
ExpandDimsstrided_slice_0:output:0ExpandDims_2/dim:output:0*
T0*,
_output_shapes
:         а[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @a
IdentityIdentitystrided_slice_0:output:0*
T0*(
_output_shapes
:         аd

Identity_1IdentityExpandDims_2:output:0*
T0*,
_output_shapes
:         а[

Identity_2IdentitySqueeze:output:0*
T0*(
_output_shapes
:         а]

Identity_3IdentitySqueeze_1:output:0*
T0*(
_output_shapes
:         аI

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: "
concat_1concat_1_0:output:0"'
concat_1_axisconcat_1/axis:output:0"#
concat_axisconcat/axis:output:0"
cudnnrnnCudnnRNN:output_h:0"!

cudnnrnn_0CudnnRNN:output_c:0"

cudnnrnn_1CudnnRNN:output:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"!

expanddimsExpandDims:output:0"%
expanddims_1ExpandDims_1:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
init_cinit_c_0"
init_hinit_h_0"/
split_1_split_dimsplit_1/split_dim:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"+
split_split_dimsplit/split_dim:output:0")
strided_slicestrided_slice_0:output:0"
	transposetranspose_0:y:0"-
transpose_1_permtranspose_1/perm:output:0"-
transpose_2_permtranspose_2/perm:output:0"-
transpose_3_permtranspose_3/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"-
transpose_5_permtranspose_5/perm:output:0"-
transpose_6_permtranspose_6/perm:output:0"-
transpose_7_permtranspose_7/perm:output:0"-
transpose_8_permtranspose_8/perm:output:0")
transpose_permtranspose/perm:output:0*(
_construction_contextkEagerRuntime*p
_input_shapes_
]:         :         а:         а:	ђ	:
ађ	:ђ	*=
api_implements+)lstm_1013b0ea-424a-45a4-86e4-0f580df52117*
api_preferred_deviceGPU*Y
backward_function_name?=__inference___backward_gpu_lstm_with_fallback_1213686_1213862*
go_backwards( *

time_major( :S O
+
_output_shapes
:         
 
_user_specified_nameinputs:PL
(
_output_shapes
:         а
 
_user_specified_nameinit_h:PL
(
_output_shapes
:         а
 
_user_specified_nameinit_c:GC

_output_shapes
:	ђ	
 
_user_specified_namekernel:RN
 
_output_shapes
:
ађ	
*
_user_specified_namerecurrent_kernel:A=

_output_shapes	
:ђ	

_user_specified_namebias
й
┴
D__inference_lstm_13_layer_call_and_return_conditional_losses_1212571
inputs_0/
read_readvariableop_resource:	ђ	2
read_1_readvariableop_resource:
ађ	-
read_2_readvariableop_resource:	ђ	

identity_3ѕбRead/ReadVariableOpбRead_1/ReadVariableOpбRead_2/ReadVariableOp=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Л
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :аs
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:         аS
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :аw
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    s
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:         аq
Read/ReadVariableOpReadVariableOpread_readvariableop_resource*
_output_shapes
:	ђ	*
dtype0[
IdentityIdentityRead/ReadVariableOp:value:0*
T0*
_output_shapes
:	ђ	v
Read_1/ReadVariableOpReadVariableOpread_1_readvariableop_resource* 
_output_shapes
:
ађ	*
dtype0`

Identity_1IdentityRead_1/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ађ	q
Read_2/ReadVariableOpReadVariableOpread_2_readvariableop_resource*
_output_shapes	
:ђ	*
dtype0[

Identity_2IdentityRead_2/ReadVariableOp:value:0*
T0*
_output_shapes	
:ђ	╣
PartitionedCallPartitionedCallinputs_0zeros:output:0zeros_1:output:0Identity:output:0Identity_1:output:0Identity_2:output:0*
Tin

2*
Tout	
2*
_collective_manager_ids
 *j
_output_shapesX
V:         а:         а:         а:         а: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ **
f%R#
!__inference_standard_lstm_1212296j

Identity_3IdentityPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         аї
NoOpNoOp^Read/ReadVariableOp^Read_1/ReadVariableOp^Read_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  : : : 2*
Read/ReadVariableOpRead/ReadVariableOp2.
Read_1/ReadVariableOpRead_1/ReadVariableOp2.
Read_2/ReadVariableOpRead_2/ReadVariableOp:^ Z
4
_output_shapes"
 :                  
"
_user_specified_name
inputs_0
к;
┴
!__inference_standard_lstm_1209725

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:         B
ShapeShapetranspose:y:0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Л
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ▓
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмє
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       Я
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУм_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ж
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maske
MatMulMatMulstrided_slice_1:output:0kernel*
T0*(
_output_shapes
:         ђ	_
MatMul_1MatMulinit_hrecurrent_kernel*
T0*(
_output_shapes
:         ђ	e
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         ђ	T
BiasAddBiasAddadd:z:0bias*
T0*(
_output_shapes
:         ђ	Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :║
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:         а:         а:         а:         а*
	num_splitU
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:         аW
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:         аT
mulMulSigmoid_1:y:0init_c*
T0*(
_output_shapes
:         аO
TanhTanhsplit:output:2*
T0*(
_output_shapes
:         аV
mul_1MulSigmoid:y:0Tanh:y:0*
T0*(
_output_shapes
:         аU
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:         аW
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:         аL
Tanh_1Tanh	add_1:z:0*
T0*(
_output_shapes
:         аZ
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*(
_output_shapes
:         аn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :┼
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Й
whileStatelessWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hinit_cstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelrecurrent_kernelbias*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*f
_output_shapesT
R: : : : :         а:         а: : :	ђ	:
ађ	:ђ	* 
_read_only_resource_inputs
 *
_stateful_parallelism( *
bodyR
while_body_1209639*
condR
while_cond_1209638*e
output_shapesT
R: : : : :         а:         а: : :	ђ	:
ађ	:ђ	*
parallel_iterations Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       О
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:         а*
element_dtype0*
num_elementsh
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ѕ
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         а*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ќ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:         а[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  ђ?a
IdentityIdentitystrided_slice_2:output:0*
T0*(
_output_shapes
:         а^

Identity_1Identitytranspose_1:y:0*
T0*,
_output_shapes
:         аY

Identity_2Identitywhile:output:4*
T0*(
_output_shapes
:         аY

Identity_3Identitywhile:output:5*
T0*(
_output_shapes
:         аI

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*p
_input_shapes_
]:         :         а:         а:	ђ	:
ађ	:ђ	*=
api_implements+)lstm_6b3f42d6-117f-4a93-968a-eb0630b50933*
api_preferred_deviceCPU*
go_backwards( *

time_major( :S O
+
_output_shapes
:         
 
_user_specified_nameinputs:PL
(
_output_shapes
:         а
 
_user_specified_nameinit_h:PL
(
_output_shapes
:         а
 
_user_specified_nameinit_c:GC

_output_shapes
:	ђ	
 
_user_specified_namekernel:RN
 
_output_shapes
:
ађ	
*
_user_specified_namerecurrent_kernel:A=

_output_shapes	
:ђ	

_user_specified_namebias
њN
б
(__forward_gpu_lstm_with_fallback_1208596

inputs
init_h_0
init_c_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4
strided_slice
cudnnrnn

cudnnrnn_0

cudnnrnn_1

cudnnrnn_2
	transpose

expanddims
expanddims_1
concat_1
transpose_perm

init_h

init_c
concat_1_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
transpose_7_perm
transpose_8_perm
split_2_split_dim
split_split_dim
split_1_split_dim
concat_axisѕc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          B
transpose_0	Transposeinputstranspose/perm:output:0*
T0P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : r

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T0*,
_output_shapes
:         аR
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : v
ExpandDims_1
ExpandDimsinit_c_0ExpandDims_1/dim:output:0*
T0*,
_output_shapes
:         аQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ї
splitSplitsplit/split_dim:output:0kernel*
T0*@
_output_shapes.
,:	а:	а:	а:	а*
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ъ
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*D
_output_shapes2
0:
аа:
аа:
аа:
аа*
	num_splite
zeros_like/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB:ђ	U
zeros_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    x

zeros_likeFill#zeros_like/shape_as_tensor:output:0zeros_like/Const:output:0*
T0*
_output_shapes	
:ђ	M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : r
concatConcatV2zeros_like:output:0biasconcat/axis:output:0*
N*
T0*
_output_shapes	
:ђS
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : Ц
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*L
_output_shapes:
8:а:а:а:а:а:а:а:а*
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
         a
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       m
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes
:	аY
ReshapeReshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:аa
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       m
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes
:	а[
	Reshape_1Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:аa
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       m
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes
:	а[
	Reshape_2Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:аa
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       m
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0*
_output_shapes
:	а[
	Reshape_3Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes	
:аa
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       p
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0* 
_output_shapes
:
аа\
	Reshape_4Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes

:ђѕa
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       p
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0* 
_output_shapes
:
аа\
	Reshape_5Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes

:ђѕa
transpose_7/permConst*
_output_shapes
:*
dtype0*
valueB"       p
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0* 
_output_shapes
:
аа\
	Reshape_6Reshapetranspose_7:y:0Const:output:0*
T0*
_output_shapes

:ђѕa
transpose_8/permConst*
_output_shapes
:*
dtype0*
valueB"       p
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0* 
_output_shapes
:
аа\
	Reshape_7Reshapetranspose_8:y:0Const:output:0*
T0*
_output_shapes

:ђѕ\
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes	
:а\
	Reshape_9Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes	
:а]

Reshape_10Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes	
:а]

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes	
:а]

Reshape_12Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes	
:а]

Reshape_13Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes	
:а]

Reshape_14Reshapesplit_2:output:6Const:output:0*
T0*
_output_shapes	
:а]

Reshape_15Reshapesplit_2:output:7Const:output:0*
T0*
_output_shapes	
:аO
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ё

concat_1_0ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T0┌
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1_0:output:0*
T0*i
_output_shapesW
U:                  а:         а:         а:f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
         _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Й
strided_slice_0StridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
shrink_axis_maskq
SqueezeSqueezeCudnnRNN:output_h:0*
T0*(
_output_shapes
:         а*
squeeze_dims
 s
	Squeeze_1SqueezeCudnnRNN:output_c:0*
T0*(
_output_shapes
:         а*
squeeze_dims
 R
ExpandDims_2/dimConst*
_output_shapes
: *
dtype0*
value	B :є
ExpandDims_2
ExpandDimsstrided_slice_0:output:0ExpandDims_2/dim:output:0*
T0*,
_output_shapes
:         а[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @a
IdentityIdentitystrided_slice_0:output:0*
T0*(
_output_shapes
:         аd

Identity_1IdentityExpandDims_2:output:0*
T0*,
_output_shapes
:         а[

Identity_2IdentitySqueeze:output:0*
T0*(
_output_shapes
:         а]

Identity_3IdentitySqueeze_1:output:0*
T0*(
_output_shapes
:         аI

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: "
concat_1concat_1_0:output:0"'
concat_1_axisconcat_1/axis:output:0"#
concat_axisconcat/axis:output:0"
cudnnrnnCudnnRNN:output_h:0"!

cudnnrnn_0CudnnRNN:output_c:0"

cudnnrnn_1CudnnRNN:output:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"!

expanddimsExpandDims:output:0"%
expanddims_1ExpandDims_1:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
init_cinit_c_0"
init_hinit_h_0"/
split_1_split_dimsplit_1/split_dim:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"+
split_split_dimsplit/split_dim:output:0")
strided_slicestrided_slice_0:output:0"
	transposetranspose_0:y:0"-
transpose_1_permtranspose_1/perm:output:0"-
transpose_2_permtranspose_2/perm:output:0"-
transpose_3_permtranspose_3/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"-
transpose_5_permtranspose_5/perm:output:0"-
transpose_6_permtranspose_6/perm:output:0"-
transpose_7_permtranspose_7/perm:output:0"-
transpose_8_permtranspose_8/perm:output:0")
transpose_permtranspose/perm:output:0*(
_construction_contextkEagerRuntime*y
_input_shapesh
f:                  :         а:         а:	ђ	:
ађ	:ђ	*=
api_implements+)lstm_d272952e-0e4d-4b32-9764-a4b66b442161*
api_preferred_deviceGPU*Y
backward_function_name?=__inference___backward_gpu_lstm_with_fallback_1208421_1208597*
go_backwards( *

time_major( :\ X
4
_output_shapes"
 :                  
 
_user_specified_nameinputs:PL
(
_output_shapes
:         а
 
_user_specified_nameinit_h:PL
(
_output_shapes
:         а
 
_user_specified_nameinit_c:GC

_output_shapes
:	ђ	
 
_user_specified_namekernel:RN
 
_output_shapes
:
ађ	
*
_user_specified_namerecurrent_kernel:A=

_output_shapes	
:ђ	

_user_specified_namebias
К
ў
*__inference_dense_13_layer_call_fn_1213873

inputs
unknown:	а
	unknown_0:
identityѕбStatefulPartitionedCall┌
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *N
fIRG
E__inference_dense_13_layer_call_and_return_conditional_losses_1209505o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         а: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         а
 
_user_specified_nameinputs
┴B
═
*__inference_gpu_lstm_with_fallback_1213254

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4ѕc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:         P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : p

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0*,
_output_shapes
:         аR
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : t
ExpandDims_1
ExpandDimsinit_cExpandDims_1/dim:output:0*
T0*,
_output_shapes
:         аQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ї
splitSplitsplit/split_dim:output:0kernel*
T0*@
_output_shapes.
,:	а:	а:	а:	а*
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ъ
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*D
_output_shapes2
0:
аа:
аа:
аа:
аа*
	num_splite
zeros_like/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB:ђ	U
zeros_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    x

zeros_likeFill#zeros_like/shape_as_tensor:output:0zeros_like/Const:output:0*
T0*
_output_shapes	
:ђ	M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : r
concatConcatV2zeros_like:output:0biasconcat/axis:output:0*
N*
T0*
_output_shapes	
:ђS
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : Ц
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*L
_output_shapes:
8:а:а:а:а:а:а:а:а*
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
         a
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       m
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes
:	аY
ReshapeReshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:аa
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       m
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes
:	а[
	Reshape_1Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:аa
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       m
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes
:	а[
	Reshape_2Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:аa
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       m
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0*
_output_shapes
:	а[
	Reshape_3Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes	
:аa
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       p
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0* 
_output_shapes
:
аа\
	Reshape_4Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes

:ђѕa
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       p
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0* 
_output_shapes
:
аа\
	Reshape_5Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes

:ђѕa
transpose_7/permConst*
_output_shapes
:*
dtype0*
valueB"       p
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0* 
_output_shapes
:
аа\
	Reshape_6Reshapetranspose_7:y:0Const:output:0*
T0*
_output_shapes

:ђѕa
transpose_8/permConst*
_output_shapes
:*
dtype0*
valueB"       p
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0* 
_output_shapes
:
аа\
	Reshape_7Reshapetranspose_8:y:0Const:output:0*
T0*
_output_shapes

:ђѕ\
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes	
:а\
	Reshape_9Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes	
:а]

Reshape_10Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes	
:а]

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes	
:а]

Reshape_12Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes	
:а]

Reshape_13Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes	
:а]

Reshape_14Reshapesplit_2:output:6Const:output:0*
T0*
_output_shapes	
:а]

Reshape_15Reshapesplit_2:output:7Const:output:0*
T0*
_output_shapes	
:аO
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : а
concat_1ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T0*
_output_shapes

:ђ╗═
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1:output:0*
T0*`
_output_shapesN
L:         а:         а:         а:f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
         _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Т
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         а*
shrink_axis_maskq
SqueezeSqueezeCudnnRNN:output_h:0*
T0*(
_output_shapes
:         а*
squeeze_dims
 s
	Squeeze_1SqueezeCudnnRNN:output_c:0*
T0*(
_output_shapes
:         а*
squeeze_dims
 R
ExpandDims_2/dimConst*
_output_shapes
: *
dtype0*
value	B :ё
ExpandDims_2
ExpandDimsstrided_slice:output:0ExpandDims_2/dim:output:0*
T0*,
_output_shapes
:         а[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @_
IdentityIdentitystrided_slice:output:0*
T0*(
_output_shapes
:         аd

Identity_1IdentityExpandDims_2:output:0*
T0*,
_output_shapes
:         а[

Identity_2IdentitySqueeze:output:0*
T0*(
_output_shapes
:         а]

Identity_3IdentitySqueeze_1:output:0*
T0*(
_output_shapes
:         аI

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*p
_input_shapes_
]:         :         а:         а:	ђ	:
ађ	:ђ	*=
api_implements+)lstm_27cab513-e281-4aef-8d50-ee58fd8ba0ea*
api_preferred_deviceGPU*
go_backwards( *

time_major( :S O
+
_output_shapes
:         
 
_user_specified_nameinputs:PL
(
_output_shapes
:         а
 
_user_specified_nameinit_h:PL
(
_output_shapes
:         а
 
_user_specified_nameinit_c:GC

_output_shapes
:	ђ	
 
_user_specified_namekernel:RN
 
_output_shapes
:
ађ	
*
_user_specified_namerecurrent_kernel:A=

_output_shapes	
:ђ	

_user_specified_namebias
Ф:
И
L__inference_change_layer_12_layer_call_and_return_conditional_losses_1210727

inputsE
2sequential_13_lstm_13_read_readvariableop_resource:	ђ	H
4sequential_13_lstm_13_read_1_readvariableop_resource:
ађ	C
4sequential_13_lstm_13_read_2_readvariableop_resource:	ђ	H
5sequential_13_dense_13_matmul_readvariableop_resource:	аD
6sequential_13_dense_13_biasadd_readvariableop_resource:
identityѕб-sequential_13/dense_13/BiasAdd/ReadVariableOpб,sequential_13/dense_13/MatMul/ReadVariableOpб)sequential_13/lstm_13/Read/ReadVariableOpб+sequential_13/lstm_13/Read_1/ReadVariableOpб+sequential_13/lstm_13/Read_2/ReadVariableOpQ
sequential_13/lstm_13/ShapeShapeinputs*
T0*
_output_shapes
:s
)sequential_13/lstm_13/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+sequential_13/lstm_13/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+sequential_13/lstm_13/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:┐
#sequential_13/lstm_13/strided_sliceStridedSlice$sequential_13/lstm_13/Shape:output:02sequential_13/lstm_13/strided_slice/stack:output:04sequential_13/lstm_13/strided_slice/stack_1:output:04sequential_13/lstm_13/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskg
$sequential_13/lstm_13/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :ах
"sequential_13/lstm_13/zeros/packedPack,sequential_13/lstm_13/strided_slice:output:0-sequential_13/lstm_13/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:f
!sequential_13/lstm_13/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    »
sequential_13/lstm_13/zerosFill+sequential_13/lstm_13/zeros/packed:output:0*sequential_13/lstm_13/zeros/Const:output:0*
T0*(
_output_shapes
:         аi
&sequential_13/lstm_13/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :а╣
$sequential_13/lstm_13/zeros_1/packedPack,sequential_13/lstm_13/strided_slice:output:0/sequential_13/lstm_13/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:h
#sequential_13/lstm_13/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    х
sequential_13/lstm_13/zeros_1Fill-sequential_13/lstm_13/zeros_1/packed:output:0,sequential_13/lstm_13/zeros_1/Const:output:0*
T0*(
_output_shapes
:         аЮ
)sequential_13/lstm_13/Read/ReadVariableOpReadVariableOp2sequential_13_lstm_13_read_readvariableop_resource*
_output_shapes
:	ђ	*
dtype0Є
sequential_13/lstm_13/IdentityIdentity1sequential_13/lstm_13/Read/ReadVariableOp:value:0*
T0*
_output_shapes
:	ђ	б
+sequential_13/lstm_13/Read_1/ReadVariableOpReadVariableOp4sequential_13_lstm_13_read_1_readvariableop_resource* 
_output_shapes
:
ађ	*
dtype0ї
 sequential_13/lstm_13/Identity_1Identity3sequential_13/lstm_13/Read_1/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ађ	Ю
+sequential_13/lstm_13/Read_2/ReadVariableOpReadVariableOp4sequential_13_lstm_13_read_2_readvariableop_resource*
_output_shapes	
:ђ	*
dtype0Є
 sequential_13/lstm_13/Identity_2Identity3sequential_13/lstm_13/Read_2/ReadVariableOp:value:0*
T0*
_output_shapes	
:ђ	╗
%sequential_13/lstm_13/PartitionedCallPartitionedCallinputs$sequential_13/lstm_13/zeros:output:0&sequential_13/lstm_13/zeros_1:output:0'sequential_13/lstm_13/Identity:output:0)sequential_13/lstm_13/Identity_1:output:0)sequential_13/lstm_13/Identity_2:output:0*
Tin

2*
Tout	
2*
_collective_manager_ids
 *j
_output_shapesX
V:         а:         а:         а:         а: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ **
f%R#
!__inference_standard_lstm_1210436Б
,sequential_13/dense_13/MatMul/ReadVariableOpReadVariableOp5sequential_13_dense_13_matmul_readvariableop_resource*
_output_shapes
:	а*
dtype0┐
sequential_13/dense_13/MatMulMatMul.sequential_13/lstm_13/PartitionedCall:output:04sequential_13/dense_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         а
-sequential_13/dense_13/BiasAdd/ReadVariableOpReadVariableOp6sequential_13_dense_13_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0╗
sequential_13/dense_13/BiasAddBiasAdd'sequential_13/dense_13/MatMul:product:05sequential_13/dense_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         u
sequential_13/reshape_13/ShapeShape'sequential_13/dense_13/BiasAdd:output:0*
T0*
_output_shapes
:v
,sequential_13/reshape_13/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: x
.sequential_13/reshape_13/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.sequential_13/reshape_13/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╬
&sequential_13/reshape_13/strided_sliceStridedSlice'sequential_13/reshape_13/Shape:output:05sequential_13/reshape_13/strided_slice/stack:output:07sequential_13/reshape_13/strided_slice/stack_1:output:07sequential_13/reshape_13/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskj
(sequential_13/reshape_13/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :j
(sequential_13/reshape_13/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :з
&sequential_13/reshape_13/Reshape/shapePack/sequential_13/reshape_13/strided_slice:output:01sequential_13/reshape_13/Reshape/shape/1:output:01sequential_13/reshape_13/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:╗
 sequential_13/reshape_13/ReshapeReshape'sequential_13/dense_13/BiasAdd:output:0/sequential_13/reshape_13/Reshape/shape:output:0*
T0*+
_output_shapes
:         u
addAddV2inputs)sequential_13/reshape_13/Reshape:output:0*
T0*+
_output_shapes
:         Z
IdentityIdentityadd:z:0^NoOp*
T0*+
_output_shapes
:         Г
NoOpNoOp.^sequential_13/dense_13/BiasAdd/ReadVariableOp-^sequential_13/dense_13/MatMul/ReadVariableOp*^sequential_13/lstm_13/Read/ReadVariableOp,^sequential_13/lstm_13/Read_1/ReadVariableOp,^sequential_13/lstm_13/Read_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:         : : : : : 2^
-sequential_13/dense_13/BiasAdd/ReadVariableOp-sequential_13/dense_13/BiasAdd/ReadVariableOp2\
,sequential_13/dense_13/MatMul/ReadVariableOp,sequential_13/dense_13/MatMul/ReadVariableOp2V
)sequential_13/lstm_13/Read/ReadVariableOp)sequential_13/lstm_13/Read/ReadVariableOp2Z
+sequential_13/lstm_13/Read_1/ReadVariableOp+sequential_13/lstm_13/Read_1/ReadVariableOp2Z
+sequential_13/lstm_13/Read_2/ReadVariableOp+sequential_13/lstm_13/Read_2/ReadVariableOp:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
к;
┴
!__inference_standard_lstm_1207873

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:         B
ShapeShapetranspose:y:0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Л
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ▓
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмє
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       Я
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУм_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ж
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maske
MatMulMatMulstrided_slice_1:output:0kernel*
T0*(
_output_shapes
:         ђ	_
MatMul_1MatMulinit_hrecurrent_kernel*
T0*(
_output_shapes
:         ђ	e
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         ђ	T
BiasAddBiasAddadd:z:0bias*
T0*(
_output_shapes
:         ђ	Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :║
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:         а:         а:         а:         а*
	num_splitU
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:         аW
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:         аT
mulMulSigmoid_1:y:0init_c*
T0*(
_output_shapes
:         аO
TanhTanhsplit:output:2*
T0*(
_output_shapes
:         аV
mul_1MulSigmoid:y:0Tanh:y:0*
T0*(
_output_shapes
:         аU
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:         аW
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:         аL
Tanh_1Tanh	add_1:z:0*
T0*(
_output_shapes
:         аZ
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*(
_output_shapes
:         аn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :┼
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Й
whileStatelessWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hinit_cstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelrecurrent_kernelbias*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*f
_output_shapesT
R: : : : :         а:         а: : :	ђ	:
ађ	:ђ	* 
_read_only_resource_inputs
 *
_stateful_parallelism( *
bodyR
while_body_1207787*
condR
while_cond_1207786*e
output_shapesT
R: : : : :         а:         а: : :	ђ	:
ађ	:ђ	*
parallel_iterations Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       О
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:         а*
element_dtype0*
num_elementsh
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ѕ
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         а*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ќ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:         а[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  ђ?a
IdentityIdentitystrided_slice_2:output:0*
T0*(
_output_shapes
:         а^

Identity_1Identitytranspose_1:y:0*
T0*,
_output_shapes
:         аY

Identity_2Identitywhile:output:4*
T0*(
_output_shapes
:         аY

Identity_3Identitywhile:output:5*
T0*(
_output_shapes
:         аI

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*p
_input_shapes_
]:         :         а:         а:	ђ	:
ађ	:ђ	*=
api_implements+)lstm_e15ea421-5ae9-42eb-bcec-970c1aedc419*
api_preferred_deviceCPU*
go_backwards( *

time_major( :S O
+
_output_shapes
:         
 
_user_specified_nameinputs:PL
(
_output_shapes
:         а
 
_user_specified_nameinit_h:PL
(
_output_shapes
:         а
 
_user_specified_nameinit_c:GC

_output_shapes
:	ђ	
 
_user_specified_namekernel:RN
 
_output_shapes
:
ађ	
*
_user_specified_namerecurrent_kernel:A=

_output_shapes	
:ђ	

_user_specified_namebias
№)
л
while_body_1210797
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_matmul_kernel_0%
!while_matmul_1_recurrent_kernel_0
while_biasadd_bias_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_matmul_kernel#
while_matmul_1_recurrent_kernel
while_biasadd_biasѕ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       д
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0њ
while/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while_matmul_kernel_0*
T0*(
_output_shapes
:         ђ	Ѓ
while/MatMul_1MatMulwhile_placeholder_2!while_matmul_1_recurrent_kernel_0*
T0*(
_output_shapes
:         ђ	w
	while/addAddV2while/MatMul:product:0while/MatMul_1:product:0*
T0*(
_output_shapes
:         ђ	p
while/BiasAddBiasAddwhile/add:z:0while_biasadd_bias_0*
T0*(
_output_shapes
:         ђ	W
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :╠
while/splitSplitwhile/split/split_dim:output:0while/BiasAdd:output:0*
T0*d
_output_shapesR
P:         а:         а:         а:         а*
	num_splita
while/SigmoidSigmoidwhile/split:output:0*
T0*(
_output_shapes
:         аc
while/Sigmoid_1Sigmoidwhile/split:output:1*
T0*(
_output_shapes
:         аm
	while/mulMulwhile/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:         а[

while/TanhTanhwhile/split:output:2*
T0*(
_output_shapes
:         аh
while/mul_1Mulwhile/Sigmoid:y:0while/Tanh:y:0*
T0*(
_output_shapes
:         аg
while/add_1AddV2while/mul:z:0while/mul_1:z:0*
T0*(
_output_shapes
:         аc
while/Sigmoid_2Sigmoidwhile/split:output:3*
T0*(
_output_shapes
:         аX
while/Tanh_1Tanhwhile/add_1:z:0*
T0*(
_output_shapes
:         аl
while/mul_2Mulwhile/Sigmoid_2:y:0while/Tanh_1:y:0*
T0*(
_output_shapes
:         аr
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : Я
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/mul_2:z:0*
_output_shapes
: *
element_dtype0:жУмO
while/add_2/yConst*
_output_shapes
: *
dtype0*
value	B :`
while/add_2AddV2while_placeholderwhile/add_2/y:output:0*
T0*
_output_shapes
: O
while/add_3/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_3AddV2while_while_loop_counterwhile/add_3/y:output:0*
T0*
_output_shapes
: L
while/IdentityIdentitywhile/add_3:z:0*
T0*
_output_shapes
: ]
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: N
while/Identity_2Identitywhile/add_2:z:0*
T0*
_output_shapes
: y
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: `
while/Identity_4Identitywhile/mul_2:z:0*
T0*(
_output_shapes
:         а`
while/Identity_5Identitywhile/add_1:z:0*
T0*(
_output_shapes
:         а"*
while_biasadd_biaswhile_biasadd_bias_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"D
while_matmul_1_recurrent_kernel!while_matmul_1_recurrent_kernel_0",
while_matmul_kernelwhile_matmul_kernel_0",
while_strided_slicewhile_strided_slice_0"е
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*e
_input_shapesT
R: : : : :         а:         а: : :	ђ	:
ађ	:ђ	: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:         а:.*
(
_output_shapes
:         а:

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	ђ	:&	"
 
_output_shapes
:
ађ	:!


_output_shapes	
:ђ	
џ	
к
while_cond_1212209
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice5
1while_while_cond_1212209___redundant_placeholder05
1while_while_cond_1212209___redundant_placeholder15
1while_while_cond_1212209___redundant_placeholder25
1while_while_cond_1212209___redundant_placeholder3
while_identity
`

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :         а:         а: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:         а:.*
(
_output_shapes
:         а:

_output_shapes
: :

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
џ	
к
while_cond_1213502
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice5
1while_while_cond_1213502___redundant_placeholder05
1while_while_cond_1213502___redundant_placeholder15
1while_while_cond_1213502___redundant_placeholder25
1while_while_cond_1213502___redundant_placeholder3
while_identity
`

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :         а:         а: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:         а:.*
(
_output_shapes
:         а:

_output_shapes
: :

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
Э
ч
/__inference_sequential_13_layer_call_fn_1210071
lstm_13_input
unknown:	ђ	
	unknown_0:
ађ	
	unknown_1:	ђ	
	unknown_2:	а
	unknown_3:
identityѕбStatefulPartitionedCallЉ
StatefulPartitionedCallStatefulPartitionedCalllstm_13_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         *'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8ѓ *S
fNRL
J__inference_sequential_13_layer_call_and_return_conditional_losses_1210043s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:         : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
+
_output_shapes
:         
'
_user_specified_namelstm_13_input
┘

c
G__inference_reshape_13_layer_call_and_return_conditional_losses_1213901

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Л
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :Ј
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:h
ReshapeReshapeinputsReshape/shape:output:0*
T0*+
_output_shapes
:         \
IdentityIdentityReshape:output:0*
T0*+
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
џ	
к
while_cond_1207786
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice5
1while_while_cond_1207786___redundant_placeholder05
1while_while_cond_1207786___redundant_placeholder15
1while_while_cond_1207786___redundant_placeholder25
1while_while_cond_1207786___redundant_placeholder3
while_identity
`

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :         а:         а: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:         а:.*
(
_output_shapes
:         а:

_output_shapes
: :

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
Э
ч
/__inference_sequential_13_layer_call_fn_1209540
lstm_13_input
unknown:	ђ	
	unknown_0:
ађ	
	unknown_1:	ђ	
	unknown_2:	а
	unknown_3:
identityѕбStatefulPartitionedCallЉ
StatefulPartitionedCallStatefulPartitionedCalllstm_13_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         *'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8ѓ *S
fNRL
J__inference_sequential_13_layer_call_and_return_conditional_losses_1209527s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:         : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
+
_output_shapes
:         
'
_user_specified_namelstm_13_input
Б
┐
D__inference_lstm_13_layer_call_and_return_conditional_losses_1213864

inputs/
read_readvariableop_resource:	ђ	2
read_1_readvariableop_resource:
ађ	-
read_2_readvariableop_resource:	ђ	

identity_3ѕбRead/ReadVariableOpбRead_1/ReadVariableOpбRead_2/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Л
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :аs
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:         аS
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :аw
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    s
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:         аq
Read/ReadVariableOpReadVariableOpread_readvariableop_resource*
_output_shapes
:	ђ	*
dtype0[
IdentityIdentityRead/ReadVariableOp:value:0*
T0*
_output_shapes
:	ђ	v
Read_1/ReadVariableOpReadVariableOpread_1_readvariableop_resource* 
_output_shapes
:
ађ	*
dtype0`

Identity_1IdentityRead_1/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ађ	q
Read_2/ReadVariableOpReadVariableOpread_2_readvariableop_resource*
_output_shapes	
:ђ	*
dtype0[

Identity_2IdentityRead_2/ReadVariableOp:value:0*
T0*
_output_shapes	
:ђ	и
PartitionedCallPartitionedCallinputszeros:output:0zeros_1:output:0Identity:output:0Identity_1:output:0Identity_2:output:0*
Tin

2*
Tout	
2*
_collective_manager_ids
 *j
_output_shapesX
V:         а:         а:         а:         а: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ **
f%R#
!__inference_standard_lstm_1213589j

Identity_3IdentityPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         аї
NoOpNoOp^Read/ReadVariableOp^Read_1/ReadVariableOp^Read_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         : : : 2*
Read/ReadVariableOpRead/ReadVariableOp2.
Read_1/ReadVariableOpRead_1/ReadVariableOp2.
Read_2/ReadVariableOpRead_2/ReadVariableOp:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
ф┬
у
=__inference___backward_gpu_lstm_with_fallback_1213686_1213862
placeholder
placeholder_1
placeholder_2
placeholder_3
placeholder_43
/gradients_expanddims_2_grad_shape_strided_slice)
%gradients_squeeze_grad_shape_cudnnrnn+
'gradients_squeeze_1_grad_shape_cudnnrnn/
+gradients_strided_slice_grad_shape_cudnnrnn!
gradients_zeros_like_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9
5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_15
1gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h,
(gradients_expanddims_1_grad_shape_init_c-
)gradients_concat_1_grad_mod_concat_1_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_permA
=gradients_transpose_7_grad_invertpermutation_transpose_7_permA
=gradients_transpose_8_grad_invertpermutation_transpose_8_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim)
%gradients_concat_grad_mod_concat_axis
identity

identity_1

identity_2

identity_3

identity_4

identity_5ѕ_
gradients/grad_ys_0Identityplaceholder*
T0*(
_output_shapes
:         аe
gradients/grad_ys_1Identityplaceholder_1*
T0*,
_output_shapes
:         аa
gradients/grad_ys_2Identityplaceholder_2*
T0*(
_output_shapes
:         аa
gradients/grad_ys_3Identityplaceholder_3*
T0*(
_output_shapes
:         аO
gradients/grad_ys_4Identityplaceholder_4*
T0*
_output_shapes
: ђ
!gradients/ExpandDims_2_grad/ShapeShape/gradients_expanddims_2_grad_shape_strided_slice*
T0*
_output_shapes
:Ф
#gradients/ExpandDims_2_grad/ReshapeReshapegradients/grad_ys_1:output:0*gradients/ExpandDims_2_grad/Shape:output:0*
T0*(
_output_shapes
:         аq
gradients/Squeeze_grad/ShapeShape%gradients_squeeze_grad_shape_cudnnrnn*
T0*
_output_shapes
:Ц
gradients/Squeeze_grad/ReshapeReshapegradients/grad_ys_2:output:0%gradients/Squeeze_grad/Shape:output:0*
T0*,
_output_shapes
:         аu
gradients/Squeeze_1_grad/ShapeShape'gradients_squeeze_1_grad_shape_cudnnrnn*
T0*
_output_shapes
:Е
 gradients/Squeeze_1_grad/ReshapeReshapegradients/grad_ys_3:output:0'gradients/Squeeze_1_grad/Shape:output:0*
T0*,
_output_shapes
:         ак
gradients/AddNAddNgradients/grad_ys_0:output:0,gradients/ExpandDims_2_grad/Reshape:output:0*
N*
T0*&
_class
loc:@gradients/grad_ys_0*(
_output_shapes
:         а}
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
:є
3gradients/strided_slice_grad/StridedSliceGrad/beginConst*
_output_shapes
:*
dtype0*
valueB:
         {
1gradients/strided_slice_grad/StridedSliceGrad/endConst*
_output_shapes
:*
dtype0*
valueB: 
5gradients/strided_slice_grad/StridedSliceGrad/stridesConst*
_output_shapes
:*
dtype0*
valueB:џ
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0<gradients/strided_slice_grad/StridedSliceGrad/begin:output:0:gradients/strided_slice_grad/StridedSliceGrad/end:output:0>gradients/strided_slice_grad/StridedSliceGrad/strides:output:0gradients/AddN:sum:0*
Index0*
T0*,
_output_shapes
:         а*
shrink_axis_maskc
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
:Ц
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_11gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1+gradients_strided_slice_grad_shape_cudnnrnn%gradients_squeeze_grad_shape_cudnnrnn'gradients_squeeze_1_grad_shape_cudnnrnn6gradients/strided_slice_grad/StridedSliceGrad:output:0'gradients/Squeeze_grad/Reshape:output:0)gradients/Squeeze_1_grad/Reshape:output:0gradients_zeros_like_cudnnrnn*
T0*c
_output_shapesQ
O:         :         а:         а:ђ╗ќ
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:л
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*+
_output_shapes
:         u
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
:к
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*(
_output_shapes
:         аy
!gradients/ExpandDims_1_grad/ShapeShape(gradients_expanddims_1_grad_shape_init_c*
T0*
_output_shapes
:╩
#gradients/ExpandDims_1_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_c_backprop:0*gradients/ExpandDims_1_grad/Shape:output:0*
T0*(
_output_shapes
:         а^
gradients/concat_1_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :џ
gradients/concat_1_grad/modFloorMod)gradients_concat_1_grad_mod_concat_1_axis%gradients/concat_1_grad/Rank:output:0*
T0*
_output_shapes
: h
gradients/concat_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:аj
gradients/concat_1_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:аj
gradients/concat_1_grad/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:аj
gradients/concat_1_grad/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:аk
gradients/concat_1_grad/Shape_4Const*
_output_shapes
:*
dtype0*
valueB:ђѕk
gradients/concat_1_grad/Shape_5Const*
_output_shapes
:*
dtype0*
valueB:ђѕk
gradients/concat_1_grad/Shape_6Const*
_output_shapes
:*
dtype0*
valueB:ђѕk
gradients/concat_1_grad/Shape_7Const*
_output_shapes
:*
dtype0*
valueB:ђѕj
gradients/concat_1_grad/Shape_8Const*
_output_shapes
:*
dtype0*
valueB:аj
gradients/concat_1_grad/Shape_9Const*
_output_shapes
:*
dtype0*
valueB:аk
 gradients/concat_1_grad/Shape_10Const*
_output_shapes
:*
dtype0*
valueB:аk
 gradients/concat_1_grad/Shape_11Const*
_output_shapes
:*
dtype0*
valueB:аk
 gradients/concat_1_grad/Shape_12Const*
_output_shapes
:*
dtype0*
valueB:аk
 gradients/concat_1_grad/Shape_13Const*
_output_shapes
:*
dtype0*
valueB:аk
 gradients/concat_1_grad/Shape_14Const*
_output_shapes
:*
dtype0*
valueB:аk
 gradients/concat_1_grad/Shape_15Const*
_output_shapes
:*
dtype0*
valueB:аЭ
$gradients/concat_1_grad/ConcatOffsetConcatOffsetgradients/concat_1_grad/mod:z:0&gradients/concat_1_grad/Shape:output:0(gradients/concat_1_grad/Shape_1:output:0(gradients/concat_1_grad/Shape_2:output:0(gradients/concat_1_grad/Shape_3:output:0(gradients/concat_1_grad/Shape_4:output:0(gradients/concat_1_grad/Shape_5:output:0(gradients/concat_1_grad/Shape_6:output:0(gradients/concat_1_grad/Shape_7:output:0(gradients/concat_1_grad/Shape_8:output:0(gradients/concat_1_grad/Shape_9:output:0)gradients/concat_1_grad/Shape_10:output:0)gradients/concat_1_grad/Shape_11:output:0)gradients/concat_1_grad/Shape_12:output:0)gradients/concat_1_grad/Shape_13:output:0)gradients/concat_1_grad/Shape_14:output:0)gradients/concat_1_grad/Shape_15:output:0*
N*t
_output_shapesb
`::::::::::::::::В
gradients/concat_1_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:0&gradients/concat_1_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:а­
gradients/concat_1_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:1(gradients/concat_1_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:а­
gradients/concat_1_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:2(gradients/concat_1_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes	
:а­
gradients/concat_1_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:3(gradients/concat_1_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes	
:аы
gradients/concat_1_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:4(gradients/concat_1_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes

:ђѕы
gradients/concat_1_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:5(gradients/concat_1_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes

:ђѕы
gradients/concat_1_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:6(gradients/concat_1_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes

:ђѕы
gradients/concat_1_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:7(gradients/concat_1_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes

:ђѕ­
gradients/concat_1_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:8(gradients/concat_1_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes	
:а­
gradients/concat_1_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:9(gradients/concat_1_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes	
:аз
 gradients/concat_1_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:10)gradients/concat_1_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes	
:аз
 gradients/concat_1_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:11)gradients/concat_1_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes	
:аз
 gradients/concat_1_grad/Slice_12Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:12)gradients/concat_1_grad/Shape_12:output:0*
Index0*
T0*
_output_shapes	
:аз
 gradients/concat_1_grad/Slice_13Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:13)gradients/concat_1_grad/Shape_13:output:0*
Index0*
T0*
_output_shapes	
:аз
 gradients/concat_1_grad/Slice_14Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:14)gradients/concat_1_grad/Shape_14:output:0*
Index0*
T0*
_output_shapes	
:аз
 gradients/concat_1_grad/Slice_15Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:15)gradients/concat_1_grad/Shape_15:output:0*
Index0*
T0*
_output_shapes	
:аm
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      б
gradients/Reshape_grad/ReshapeReshape&gradients/concat_1_grad/Slice:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes
:	аo
gradients/Reshape_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      е
 gradients/Reshape_1_grad/ReshapeReshape(gradients/concat_1_grad/Slice_1:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0*
_output_shapes
:	аo
gradients/Reshape_2_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      е
 gradients/Reshape_2_grad/ReshapeReshape(gradients/concat_1_grad/Slice_2:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0*
_output_shapes
:	аo
gradients/Reshape_3_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      е
 gradients/Reshape_3_grad/ReshapeReshape(gradients/concat_1_grad/Slice_3:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0*
_output_shapes
:	аo
gradients/Reshape_4_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      Е
 gradients/Reshape_4_grad/ReshapeReshape(gradients/concat_1_grad/Slice_4:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0* 
_output_shapes
:
ааo
gradients/Reshape_5_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      Е
 gradients/Reshape_5_grad/ReshapeReshape(gradients/concat_1_grad/Slice_5:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0* 
_output_shapes
:
ааo
gradients/Reshape_6_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      Е
 gradients/Reshape_6_grad/ReshapeReshape(gradients/concat_1_grad/Slice_6:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0* 
_output_shapes
:
ааo
gradients/Reshape_7_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      Е
 gradients/Reshape_7_grad/ReshapeReshape(gradients/concat_1_grad/Slice_7:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0* 
_output_shapes
:
ааi
gradients/Reshape_8_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:ац
 gradients/Reshape_8_grad/ReshapeReshape(gradients/concat_1_grad/Slice_8:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes	
:аi
gradients/Reshape_9_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:ац
 gradients/Reshape_9_grad/ReshapeReshape(gradients/concat_1_grad/Slice_9:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes	
:аj
gradients/Reshape_10_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:аД
!gradients/Reshape_10_grad/ReshapeReshape)gradients/concat_1_grad/Slice_10:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes	
:аj
gradients/Reshape_11_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:аД
!gradients/Reshape_11_grad/ReshapeReshape)gradients/concat_1_grad/Slice_11:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes	
:аj
gradients/Reshape_12_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:аД
!gradients/Reshape_12_grad/ReshapeReshape)gradients/concat_1_grad/Slice_12:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes	
:аj
gradients/Reshape_13_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:аД
!gradients/Reshape_13_grad/ReshapeReshape)gradients/concat_1_grad/Slice_13:output:0(gradients/Reshape_13_grad/Shape:output:0*
T0*
_output_shapes	
:аj
gradients/Reshape_14_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:аД
!gradients/Reshape_14_grad/ReshapeReshape)gradients/concat_1_grad/Slice_14:output:0(gradients/Reshape_14_grad/Shape:output:0*
T0*
_output_shapes	
:аj
gradients/Reshape_15_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:аД
!gradients/Reshape_15_grad/ReshapeReshape)gradients/concat_1_grad/Slice_15:output:0(gradients/Reshape_15_grad/Shape:output:0*
T0*
_output_shapes	
:аю
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:Х
$gradients/transpose_1_grad/transpose	Transpose'gradients/Reshape_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0*
_output_shapes
:	аю
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:И
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0*
_output_shapes
:	аю
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:И
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0*
_output_shapes
:	аю
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:И
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0*
_output_shapes
:	аю
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:╣
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
ааю
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:╣
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
ааю
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:╣
$gradients/transpose_7_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
ааю
,gradients/transpose_8_grad/InvertPermutationInvertPermutation=gradients_transpose_8_grad_invertpermutation_transpose_8_perm*
_output_shapes
:╣
$gradients/transpose_8_grad/transpose	Transpose)gradients/Reshape_7_grad/Reshape:output:00gradients/transpose_8_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
аау
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0*gradients/Reshape_13_grad/Reshape:output:0*gradients/Reshape_14_grad/Reshape:output:0*gradients/Reshape_15_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
N*
T0*
_output_shapes	
:ђ»
gradients/split_grad/concatConcatV2(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
N*
T0*
_output_shapes
:	ђ	Х
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0(gradients/transpose_7_grad/transpose:y:0(gradients/transpose_8_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
N*
T0* 
_output_shapes
:
ађ	\
gradients/concat_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :њ
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: f
gradients/concat_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:ђ	h
gradients/concat_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:ђ	╩
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0*
N* 
_output_shapes
::м
gradients/concat_grad/SliceSlice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:ђ	о
gradients/concat_grad/Slice_1Slice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:ђ	r
IdentityIdentity&gradients/transpose_grad/transpose:y:0*
T0*+
_output_shapes
:         u

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0*
T0*(
_output_shapes
:         аw

Identity_2Identity,gradients/ExpandDims_1_grad/Reshape:output:0*
T0*(
_output_shapes
:         аf

Identity_3Identity$gradients/split_grad/concat:output:0*
T0*
_output_shapes
:	ђ	i

Identity_4Identity&gradients/split_1_grad/concat:output:0*
T0* 
_output_shapes
:
ађ	d

Identity_5Identity&gradients/concat_grad/Slice_1:output:0*
T0*
_output_shapes	
:ђ	"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*(
_construction_contextkEagerRuntime*ѓ
_input_shapes­
ь:         а:         а:         а:         а: :         а:         а:         а:         а::         :         а:         а:ђ╗::         а:         а: ::::::::: : : : *=
api_implements+)lstm_1013b0ea-424a-45a4-86e4-0f580df52117*
api_preferred_deviceGPU*C
forward_function_name*(__forward_gpu_lstm_with_fallback_1213861*
go_backwards( *

time_major( :. *
(
_output_shapes
:         а:2.
,
_output_shapes
:         а:.*
(
_output_shapes
:         а:.*
(
_output_shapes
:         а:

_output_shapes
: :.*
(
_output_shapes
:         а:2.
,
_output_shapes
:         а:2.
,
_output_shapes
:         а:2.
,
_output_shapes
:         а:	

_output_shapes
::1
-
+
_output_shapes
:         :2.
,
_output_shapes
:         а:2.
,
_output_shapes
:         а:"

_output_shapes

:ђ╗: 

_output_shapes
::.*
(
_output_shapes
:         а:.*
(
_output_shapes
:         а:

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ф┬
у
=__inference___backward_gpu_lstm_with_fallback_1211903_1212079
placeholder
placeholder_1
placeholder_2
placeholder_3
placeholder_43
/gradients_expanddims_2_grad_shape_strided_slice)
%gradients_squeeze_grad_shape_cudnnrnn+
'gradients_squeeze_1_grad_shape_cudnnrnn/
+gradients_strided_slice_grad_shape_cudnnrnn!
gradients_zeros_like_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9
5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_15
1gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h,
(gradients_expanddims_1_grad_shape_init_c-
)gradients_concat_1_grad_mod_concat_1_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_permA
=gradients_transpose_7_grad_invertpermutation_transpose_7_permA
=gradients_transpose_8_grad_invertpermutation_transpose_8_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim)
%gradients_concat_grad_mod_concat_axis
identity

identity_1

identity_2

identity_3

identity_4

identity_5ѕ_
gradients/grad_ys_0Identityplaceholder*
T0*(
_output_shapes
:         аe
gradients/grad_ys_1Identityplaceholder_1*
T0*,
_output_shapes
:         аa
gradients/grad_ys_2Identityplaceholder_2*
T0*(
_output_shapes
:         аa
gradients/grad_ys_3Identityplaceholder_3*
T0*(
_output_shapes
:         аO
gradients/grad_ys_4Identityplaceholder_4*
T0*
_output_shapes
: ђ
!gradients/ExpandDims_2_grad/ShapeShape/gradients_expanddims_2_grad_shape_strided_slice*
T0*
_output_shapes
:Ф
#gradients/ExpandDims_2_grad/ReshapeReshapegradients/grad_ys_1:output:0*gradients/ExpandDims_2_grad/Shape:output:0*
T0*(
_output_shapes
:         аq
gradients/Squeeze_grad/ShapeShape%gradients_squeeze_grad_shape_cudnnrnn*
T0*
_output_shapes
:Ц
gradients/Squeeze_grad/ReshapeReshapegradients/grad_ys_2:output:0%gradients/Squeeze_grad/Shape:output:0*
T0*,
_output_shapes
:         аu
gradients/Squeeze_1_grad/ShapeShape'gradients_squeeze_1_grad_shape_cudnnrnn*
T0*
_output_shapes
:Е
 gradients/Squeeze_1_grad/ReshapeReshapegradients/grad_ys_3:output:0'gradients/Squeeze_1_grad/Shape:output:0*
T0*,
_output_shapes
:         ак
gradients/AddNAddNgradients/grad_ys_0:output:0,gradients/ExpandDims_2_grad/Reshape:output:0*
N*
T0*&
_class
loc:@gradients/grad_ys_0*(
_output_shapes
:         а}
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
:є
3gradients/strided_slice_grad/StridedSliceGrad/beginConst*
_output_shapes
:*
dtype0*
valueB:
         {
1gradients/strided_slice_grad/StridedSliceGrad/endConst*
_output_shapes
:*
dtype0*
valueB: 
5gradients/strided_slice_grad/StridedSliceGrad/stridesConst*
_output_shapes
:*
dtype0*
valueB:џ
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0<gradients/strided_slice_grad/StridedSliceGrad/begin:output:0:gradients/strided_slice_grad/StridedSliceGrad/end:output:0>gradients/strided_slice_grad/StridedSliceGrad/strides:output:0gradients/AddN:sum:0*
Index0*
T0*,
_output_shapes
:         а*
shrink_axis_maskc
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
:Ц
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_11gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1+gradients_strided_slice_grad_shape_cudnnrnn%gradients_squeeze_grad_shape_cudnnrnn'gradients_squeeze_1_grad_shape_cudnnrnn6gradients/strided_slice_grad/StridedSliceGrad:output:0'gradients/Squeeze_grad/Reshape:output:0)gradients/Squeeze_1_grad/Reshape:output:0gradients_zeros_like_cudnnrnn*
T0*c
_output_shapesQ
O:         :         а:         а:ђ╗ќ
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:л
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*+
_output_shapes
:         u
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
:к
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*(
_output_shapes
:         аy
!gradients/ExpandDims_1_grad/ShapeShape(gradients_expanddims_1_grad_shape_init_c*
T0*
_output_shapes
:╩
#gradients/ExpandDims_1_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_c_backprop:0*gradients/ExpandDims_1_grad/Shape:output:0*
T0*(
_output_shapes
:         а^
gradients/concat_1_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :џ
gradients/concat_1_grad/modFloorMod)gradients_concat_1_grad_mod_concat_1_axis%gradients/concat_1_grad/Rank:output:0*
T0*
_output_shapes
: h
gradients/concat_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:аj
gradients/concat_1_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:аj
gradients/concat_1_grad/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:аj
gradients/concat_1_grad/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:аk
gradients/concat_1_grad/Shape_4Const*
_output_shapes
:*
dtype0*
valueB:ђѕk
gradients/concat_1_grad/Shape_5Const*
_output_shapes
:*
dtype0*
valueB:ђѕk
gradients/concat_1_grad/Shape_6Const*
_output_shapes
:*
dtype0*
valueB:ђѕk
gradients/concat_1_grad/Shape_7Const*
_output_shapes
:*
dtype0*
valueB:ђѕj
gradients/concat_1_grad/Shape_8Const*
_output_shapes
:*
dtype0*
valueB:аj
gradients/concat_1_grad/Shape_9Const*
_output_shapes
:*
dtype0*
valueB:аk
 gradients/concat_1_grad/Shape_10Const*
_output_shapes
:*
dtype0*
valueB:аk
 gradients/concat_1_grad/Shape_11Const*
_output_shapes
:*
dtype0*
valueB:аk
 gradients/concat_1_grad/Shape_12Const*
_output_shapes
:*
dtype0*
valueB:аk
 gradients/concat_1_grad/Shape_13Const*
_output_shapes
:*
dtype0*
valueB:аk
 gradients/concat_1_grad/Shape_14Const*
_output_shapes
:*
dtype0*
valueB:аk
 gradients/concat_1_grad/Shape_15Const*
_output_shapes
:*
dtype0*
valueB:аЭ
$gradients/concat_1_grad/ConcatOffsetConcatOffsetgradients/concat_1_grad/mod:z:0&gradients/concat_1_grad/Shape:output:0(gradients/concat_1_grad/Shape_1:output:0(gradients/concat_1_grad/Shape_2:output:0(gradients/concat_1_grad/Shape_3:output:0(gradients/concat_1_grad/Shape_4:output:0(gradients/concat_1_grad/Shape_5:output:0(gradients/concat_1_grad/Shape_6:output:0(gradients/concat_1_grad/Shape_7:output:0(gradients/concat_1_grad/Shape_8:output:0(gradients/concat_1_grad/Shape_9:output:0)gradients/concat_1_grad/Shape_10:output:0)gradients/concat_1_grad/Shape_11:output:0)gradients/concat_1_grad/Shape_12:output:0)gradients/concat_1_grad/Shape_13:output:0)gradients/concat_1_grad/Shape_14:output:0)gradients/concat_1_grad/Shape_15:output:0*
N*t
_output_shapesb
`::::::::::::::::В
gradients/concat_1_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:0&gradients/concat_1_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:а­
gradients/concat_1_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:1(gradients/concat_1_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:а­
gradients/concat_1_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:2(gradients/concat_1_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes	
:а­
gradients/concat_1_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:3(gradients/concat_1_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes	
:аы
gradients/concat_1_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:4(gradients/concat_1_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes

:ђѕы
gradients/concat_1_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:5(gradients/concat_1_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes

:ђѕы
gradients/concat_1_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:6(gradients/concat_1_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes

:ђѕы
gradients/concat_1_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:7(gradients/concat_1_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes

:ђѕ­
gradients/concat_1_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:8(gradients/concat_1_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes	
:а­
gradients/concat_1_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:9(gradients/concat_1_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes	
:аз
 gradients/concat_1_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:10)gradients/concat_1_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes	
:аз
 gradients/concat_1_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:11)gradients/concat_1_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes	
:аз
 gradients/concat_1_grad/Slice_12Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:12)gradients/concat_1_grad/Shape_12:output:0*
Index0*
T0*
_output_shapes	
:аз
 gradients/concat_1_grad/Slice_13Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:13)gradients/concat_1_grad/Shape_13:output:0*
Index0*
T0*
_output_shapes	
:аз
 gradients/concat_1_grad/Slice_14Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:14)gradients/concat_1_grad/Shape_14:output:0*
Index0*
T0*
_output_shapes	
:аз
 gradients/concat_1_grad/Slice_15Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:15)gradients/concat_1_grad/Shape_15:output:0*
Index0*
T0*
_output_shapes	
:аm
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      б
gradients/Reshape_grad/ReshapeReshape&gradients/concat_1_grad/Slice:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes
:	аo
gradients/Reshape_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      е
 gradients/Reshape_1_grad/ReshapeReshape(gradients/concat_1_grad/Slice_1:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0*
_output_shapes
:	аo
gradients/Reshape_2_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      е
 gradients/Reshape_2_grad/ReshapeReshape(gradients/concat_1_grad/Slice_2:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0*
_output_shapes
:	аo
gradients/Reshape_3_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      е
 gradients/Reshape_3_grad/ReshapeReshape(gradients/concat_1_grad/Slice_3:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0*
_output_shapes
:	аo
gradients/Reshape_4_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      Е
 gradients/Reshape_4_grad/ReshapeReshape(gradients/concat_1_grad/Slice_4:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0* 
_output_shapes
:
ааo
gradients/Reshape_5_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      Е
 gradients/Reshape_5_grad/ReshapeReshape(gradients/concat_1_grad/Slice_5:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0* 
_output_shapes
:
ааo
gradients/Reshape_6_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      Е
 gradients/Reshape_6_grad/ReshapeReshape(gradients/concat_1_grad/Slice_6:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0* 
_output_shapes
:
ааo
gradients/Reshape_7_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      Е
 gradients/Reshape_7_grad/ReshapeReshape(gradients/concat_1_grad/Slice_7:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0* 
_output_shapes
:
ааi
gradients/Reshape_8_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:ац
 gradients/Reshape_8_grad/ReshapeReshape(gradients/concat_1_grad/Slice_8:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes	
:аi
gradients/Reshape_9_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:ац
 gradients/Reshape_9_grad/ReshapeReshape(gradients/concat_1_grad/Slice_9:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes	
:аj
gradients/Reshape_10_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:аД
!gradients/Reshape_10_grad/ReshapeReshape)gradients/concat_1_grad/Slice_10:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes	
:аj
gradients/Reshape_11_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:аД
!gradients/Reshape_11_grad/ReshapeReshape)gradients/concat_1_grad/Slice_11:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes	
:аj
gradients/Reshape_12_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:аД
!gradients/Reshape_12_grad/ReshapeReshape)gradients/concat_1_grad/Slice_12:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes	
:аj
gradients/Reshape_13_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:аД
!gradients/Reshape_13_grad/ReshapeReshape)gradients/concat_1_grad/Slice_13:output:0(gradients/Reshape_13_grad/Shape:output:0*
T0*
_output_shapes	
:аj
gradients/Reshape_14_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:аД
!gradients/Reshape_14_grad/ReshapeReshape)gradients/concat_1_grad/Slice_14:output:0(gradients/Reshape_14_grad/Shape:output:0*
T0*
_output_shapes	
:аj
gradients/Reshape_15_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:аД
!gradients/Reshape_15_grad/ReshapeReshape)gradients/concat_1_grad/Slice_15:output:0(gradients/Reshape_15_grad/Shape:output:0*
T0*
_output_shapes	
:аю
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:Х
$gradients/transpose_1_grad/transpose	Transpose'gradients/Reshape_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0*
_output_shapes
:	аю
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:И
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0*
_output_shapes
:	аю
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:И
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0*
_output_shapes
:	аю
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:И
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0*
_output_shapes
:	аю
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:╣
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
ааю
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:╣
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
ааю
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:╣
$gradients/transpose_7_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
ааю
,gradients/transpose_8_grad/InvertPermutationInvertPermutation=gradients_transpose_8_grad_invertpermutation_transpose_8_perm*
_output_shapes
:╣
$gradients/transpose_8_grad/transpose	Transpose)gradients/Reshape_7_grad/Reshape:output:00gradients/transpose_8_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
аау
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0*gradients/Reshape_13_grad/Reshape:output:0*gradients/Reshape_14_grad/Reshape:output:0*gradients/Reshape_15_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
N*
T0*
_output_shapes	
:ђ»
gradients/split_grad/concatConcatV2(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
N*
T0*
_output_shapes
:	ђ	Х
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0(gradients/transpose_7_grad/transpose:y:0(gradients/transpose_8_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
N*
T0* 
_output_shapes
:
ађ	\
gradients/concat_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :њ
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: f
gradients/concat_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:ђ	h
gradients/concat_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:ђ	╩
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0*
N* 
_output_shapes
::м
gradients/concat_grad/SliceSlice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:ђ	о
gradients/concat_grad/Slice_1Slice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:ђ	r
IdentityIdentity&gradients/transpose_grad/transpose:y:0*
T0*+
_output_shapes
:         u

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0*
T0*(
_output_shapes
:         аw

Identity_2Identity,gradients/ExpandDims_1_grad/Reshape:output:0*
T0*(
_output_shapes
:         аf

Identity_3Identity$gradients/split_grad/concat:output:0*
T0*
_output_shapes
:	ђ	i

Identity_4Identity&gradients/split_1_grad/concat:output:0*
T0* 
_output_shapes
:
ађ	d

Identity_5Identity&gradients/concat_grad/Slice_1:output:0*
T0*
_output_shapes	
:ђ	"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*(
_construction_contextkEagerRuntime*ѓ
_input_shapes­
ь:         а:         а:         а:         а: :         а:         а:         а:         а::         :         а:         а:ђ╗::         а:         а: ::::::::: : : : *=
api_implements+)lstm_f0644406-b4c4-48ed-b733-8b83744a8936*
api_preferred_deviceGPU*C
forward_function_name*(__forward_gpu_lstm_with_fallback_1212078*
go_backwards( *

time_major( :. *
(
_output_shapes
:         а:2.
,
_output_shapes
:         а:.*
(
_output_shapes
:         а:.*
(
_output_shapes
:         а:

_output_shapes
: :.*
(
_output_shapes
:         а:2.
,
_output_shapes
:         а:2.
,
_output_shapes
:         а:2.
,
_output_shapes
:         а:	

_output_shapes
::1
-
+
_output_shapes
:         :2.
,
_output_shapes
:         а:2.
,
_output_shapes
:         а:"

_output_shapes

:ђ╗: 

_output_shapes
::.*
(
_output_shapes
:         а:.*
(
_output_shapes
:         а:

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
┴B
═
*__inference_gpu_lstm_with_fallback_1211902

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4ѕc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:         P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : p

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0*,
_output_shapes
:         аR
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : t
ExpandDims_1
ExpandDimsinit_cExpandDims_1/dim:output:0*
T0*,
_output_shapes
:         аQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ї
splitSplitsplit/split_dim:output:0kernel*
T0*@
_output_shapes.
,:	а:	а:	а:	а*
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ъ
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*D
_output_shapes2
0:
аа:
аа:
аа:
аа*
	num_splite
zeros_like/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB:ђ	U
zeros_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    x

zeros_likeFill#zeros_like/shape_as_tensor:output:0zeros_like/Const:output:0*
T0*
_output_shapes	
:ђ	M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : r
concatConcatV2zeros_like:output:0biasconcat/axis:output:0*
N*
T0*
_output_shapes	
:ђS
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : Ц
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*L
_output_shapes:
8:а:а:а:а:а:а:а:а*
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
         a
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       m
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes
:	аY
ReshapeReshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:аa
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       m
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes
:	а[
	Reshape_1Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:аa
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       m
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes
:	а[
	Reshape_2Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:аa
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       m
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0*
_output_shapes
:	а[
	Reshape_3Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes	
:аa
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       p
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0* 
_output_shapes
:
аа\
	Reshape_4Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes

:ђѕa
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       p
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0* 
_output_shapes
:
аа\
	Reshape_5Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes

:ђѕa
transpose_7/permConst*
_output_shapes
:*
dtype0*
valueB"       p
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0* 
_output_shapes
:
аа\
	Reshape_6Reshapetranspose_7:y:0Const:output:0*
T0*
_output_shapes

:ђѕa
transpose_8/permConst*
_output_shapes
:*
dtype0*
valueB"       p
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0* 
_output_shapes
:
аа\
	Reshape_7Reshapetranspose_8:y:0Const:output:0*
T0*
_output_shapes

:ђѕ\
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes	
:а\
	Reshape_9Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes	
:а]

Reshape_10Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes	
:а]

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes	
:а]

Reshape_12Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes	
:а]

Reshape_13Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes	
:а]

Reshape_14Reshapesplit_2:output:6Const:output:0*
T0*
_output_shapes	
:а]

Reshape_15Reshapesplit_2:output:7Const:output:0*
T0*
_output_shapes	
:аO
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : а
concat_1ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T0*
_output_shapes

:ђ╗═
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1:output:0*
T0*`
_output_shapesN
L:         а:         а:         а:f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
         _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Т
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         а*
shrink_axis_maskq
SqueezeSqueezeCudnnRNN:output_h:0*
T0*(
_output_shapes
:         а*
squeeze_dims
 s
	Squeeze_1SqueezeCudnnRNN:output_c:0*
T0*(
_output_shapes
:         а*
squeeze_dims
 R
ExpandDims_2/dimConst*
_output_shapes
: *
dtype0*
value	B :ё
ExpandDims_2
ExpandDimsstrided_slice:output:0ExpandDims_2/dim:output:0*
T0*,
_output_shapes
:         а[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @_
IdentityIdentitystrided_slice:output:0*
T0*(
_output_shapes
:         аd

Identity_1IdentityExpandDims_2:output:0*
T0*,
_output_shapes
:         а[

Identity_2IdentitySqueeze:output:0*
T0*(
_output_shapes
:         а]

Identity_3IdentitySqueeze_1:output:0*
T0*(
_output_shapes
:         аI

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*p
_input_shapes_
]:         :         а:         а:	ђ	:
ађ	:ђ	*=
api_implements+)lstm_f0644406-b4c4-48ed-b733-8b83744a8936*
api_preferred_deviceGPU*
go_backwards( *

time_major( :S O
+
_output_shapes
:         
 
_user_specified_nameinputs:PL
(
_output_shapes
:         а
 
_user_specified_nameinit_h:PL
(
_output_shapes
:         а
 
_user_specified_nameinit_c:GC

_output_shapes
:	ђ	
 
_user_specified_namekernel:RN
 
_output_shapes
:
ађ	
*
_user_specified_namerecurrent_kernel:A=

_output_shapes	
:ђ	

_user_specified_namebias
№)
л
while_body_1208680
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_matmul_kernel_0%
!while_matmul_1_recurrent_kernel_0
while_biasadd_bias_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_matmul_kernel#
while_matmul_1_recurrent_kernel
while_biasadd_biasѕ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       д
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0њ
while/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while_matmul_kernel_0*
T0*(
_output_shapes
:         ђ	Ѓ
while/MatMul_1MatMulwhile_placeholder_2!while_matmul_1_recurrent_kernel_0*
T0*(
_output_shapes
:         ђ	w
	while/addAddV2while/MatMul:product:0while/MatMul_1:product:0*
T0*(
_output_shapes
:         ђ	p
while/BiasAddBiasAddwhile/add:z:0while_biasadd_bias_0*
T0*(
_output_shapes
:         ђ	W
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :╠
while/splitSplitwhile/split/split_dim:output:0while/BiasAdd:output:0*
T0*d
_output_shapesR
P:         а:         а:         а:         а*
	num_splita
while/SigmoidSigmoidwhile/split:output:0*
T0*(
_output_shapes
:         аc
while/Sigmoid_1Sigmoidwhile/split:output:1*
T0*(
_output_shapes
:         аm
	while/mulMulwhile/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:         а[

while/TanhTanhwhile/split:output:2*
T0*(
_output_shapes
:         аh
while/mul_1Mulwhile/Sigmoid:y:0while/Tanh:y:0*
T0*(
_output_shapes
:         аg
while/add_1AddV2while/mul:z:0while/mul_1:z:0*
T0*(
_output_shapes
:         аc
while/Sigmoid_2Sigmoidwhile/split:output:3*
T0*(
_output_shapes
:         аX
while/Tanh_1Tanhwhile/add_1:z:0*
T0*(
_output_shapes
:         аl
while/mul_2Mulwhile/Sigmoid_2:y:0while/Tanh_1:y:0*
T0*(
_output_shapes
:         аr
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : Я
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/mul_2:z:0*
_output_shapes
: *
element_dtype0:жУмO
while/add_2/yConst*
_output_shapes
: *
dtype0*
value	B :`
while/add_2AddV2while_placeholderwhile/add_2/y:output:0*
T0*
_output_shapes
: O
while/add_3/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_3AddV2while_while_loop_counterwhile/add_3/y:output:0*
T0*
_output_shapes
: L
while/IdentityIdentitywhile/add_3:z:0*
T0*
_output_shapes
: ]
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: N
while/Identity_2Identitywhile/add_2:z:0*
T0*
_output_shapes
: y
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: `
while/Identity_4Identitywhile/mul_2:z:0*
T0*(
_output_shapes
:         а`
while/Identity_5Identitywhile/add_1:z:0*
T0*(
_output_shapes
:         а"*
while_biasadd_biaswhile_biasadd_bias_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"D
while_matmul_1_recurrent_kernel!while_matmul_1_recurrent_kernel_0",
while_matmul_kernelwhile_matmul_kernel_0",
while_strided_slicewhile_strided_slice_0"е
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*e
_input_shapesT
R: : : : :         а:         а: : :	ђ	:
ађ	:ђ	: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:         а:.*
(
_output_shapes
:         а:

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	ђ	:&	"
 
_output_shapes
:
ађ	:!


_output_shapes	
:ђ	
х
┐
D__inference_lstm_13_layer_call_and_return_conditional_losses_1208599

inputs/
read_readvariableop_resource:	ђ	2
read_1_readvariableop_resource:
ађ	-
read_2_readvariableop_resource:	ђ	

identity_3ѕбRead/ReadVariableOpбRead_1/ReadVariableOpбRead_2/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Л
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :аs
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:         аS
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :аw
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    s
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:         аq
Read/ReadVariableOpReadVariableOpread_readvariableop_resource*
_output_shapes
:	ђ	*
dtype0[
IdentityIdentityRead/ReadVariableOp:value:0*
T0*
_output_shapes
:	ђ	v
Read_1/ReadVariableOpReadVariableOpread_1_readvariableop_resource* 
_output_shapes
:
ађ	*
dtype0`

Identity_1IdentityRead_1/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ађ	q
Read_2/ReadVariableOpReadVariableOpread_2_readvariableop_resource*
_output_shapes	
:ђ	*
dtype0[

Identity_2IdentityRead_2/ReadVariableOp:value:0*
T0*
_output_shapes	
:ђ	и
PartitionedCallPartitionedCallinputszeros:output:0zeros_1:output:0Identity:output:0Identity_1:output:0Identity_2:output:0*
Tin

2*
Tout	
2*
_collective_manager_ids
 *j
_output_shapesX
V:         а:         а:         а:         а: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ **
f%R#
!__inference_standard_lstm_1208324j

Identity_3IdentityPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         аї
NoOpNoOp^Read/ReadVariableOp^Read_1/ReadVariableOp^Read_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  : : : 2*
Read/ReadVariableOpRead/ReadVariableOp2.
Read_1/ReadVariableOpRead_1/ReadVariableOp2.
Read_2/ReadVariableOpRead_2/ReadVariableOp:\ X
4
_output_shapes"
 :                  
 
_user_specified_nameinputs
к;
┴
!__inference_standard_lstm_1213158

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:         B
ShapeShapetranspose:y:0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Л
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ▓
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмє
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       Я
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУм_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ж
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maske
MatMulMatMulstrided_slice_1:output:0kernel*
T0*(
_output_shapes
:         ђ	_
MatMul_1MatMulinit_hrecurrent_kernel*
T0*(
_output_shapes
:         ђ	e
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         ђ	T
BiasAddBiasAddadd:z:0bias*
T0*(
_output_shapes
:         ђ	Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :║
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:         а:         а:         а:         а*
	num_splitU
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:         аW
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:         аT
mulMulSigmoid_1:y:0init_c*
T0*(
_output_shapes
:         аO
TanhTanhsplit:output:2*
T0*(
_output_shapes
:         аV
mul_1MulSigmoid:y:0Tanh:y:0*
T0*(
_output_shapes
:         аU
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:         аW
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:         аL
Tanh_1Tanh	add_1:z:0*
T0*(
_output_shapes
:         аZ
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*(
_output_shapes
:         аn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :┼
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Й
whileStatelessWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hinit_cstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelrecurrent_kernelbias*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*f
_output_shapesT
R: : : : :         а:         а: : :	ђ	:
ађ	:ђ	* 
_read_only_resource_inputs
 *
_stateful_parallelism( *
bodyR
while_body_1213072*
condR
while_cond_1213071*e
output_shapesT
R: : : : :         а:         а: : :	ђ	:
ађ	:ђ	*
parallel_iterations Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       О
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:         а*
element_dtype0*
num_elementsh
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ѕ
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         а*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ќ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:         а[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  ђ?a
IdentityIdentitystrided_slice_2:output:0*
T0*(
_output_shapes
:         а^

Identity_1Identitytranspose_1:y:0*
T0*,
_output_shapes
:         аY

Identity_2Identitywhile:output:4*
T0*(
_output_shapes
:         аY

Identity_3Identitywhile:output:5*
T0*(
_output_shapes
:         аI

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*p
_input_shapes_
]:         :         а:         а:	ђ	:
ађ	:ђ	*=
api_implements+)lstm_27cab513-e281-4aef-8d50-ee58fd8ba0ea*
api_preferred_deviceCPU*
go_backwards( *

time_major( :S O
+
_output_shapes
:         
 
_user_specified_nameinputs:PL
(
_output_shapes
:         а
 
_user_specified_nameinit_h:PL
(
_output_shapes
:         а
 
_user_specified_nameinit_c:GC

_output_shapes
:	ђ	
 
_user_specified_namekernel:RN
 
_output_shapes
:
ађ	
*
_user_specified_namerecurrent_kernel:A=

_output_shapes	
:ђ	

_user_specified_namebias
р;
┴
!__inference_standard_lstm_1208324

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :                  B
ShapeShapetranspose:y:0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Л
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ▓
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмє
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       Я
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУм_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ж
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maske
MatMulMatMulstrided_slice_1:output:0kernel*
T0*(
_output_shapes
:         ђ	_
MatMul_1MatMulinit_hrecurrent_kernel*
T0*(
_output_shapes
:         ђ	e
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         ђ	T
BiasAddBiasAddadd:z:0bias*
T0*(
_output_shapes
:         ђ	Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :║
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:         а:         а:         а:         а*
	num_splitU
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:         аW
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:         аT
mulMulSigmoid_1:y:0init_c*
T0*(
_output_shapes
:         аO
TanhTanhsplit:output:2*
T0*(
_output_shapes
:         аV
mul_1MulSigmoid:y:0Tanh:y:0*
T0*(
_output_shapes
:         аU
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:         аW
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:         аL
Tanh_1Tanh	add_1:z:0*
T0*(
_output_shapes
:         аZ
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*(
_output_shapes
:         аn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :┼
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Й
whileStatelessWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hinit_cstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelrecurrent_kernelbias*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*f
_output_shapesT
R: : : : :         а:         а: : :	ђ	:
ађ	:ђ	* 
_read_only_resource_inputs
 *
_stateful_parallelism( *
bodyR
while_body_1208238*
condR
while_cond_1208237*e
output_shapesT
R: : : : :         а:         а: : :	ђ	:
ађ	:ђ	*
parallel_iterations Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       О
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:         а*
element_dtype0*
num_elementsh
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ѕ
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         а*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ќ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:         а[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  ђ?a
IdentityIdentitystrided_slice_2:output:0*
T0*(
_output_shapes
:         а^

Identity_1Identitytranspose_1:y:0*
T0*,
_output_shapes
:         аY

Identity_2Identitywhile:output:4*
T0*(
_output_shapes
:         аY

Identity_3Identitywhile:output:5*
T0*(
_output_shapes
:         аI

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*y
_input_shapesh
f:                  :         а:         а:	ђ	:
ађ	:ђ	*=
api_implements+)lstm_d272952e-0e4d-4b32-9764-a4b66b442161*
api_preferred_deviceCPU*
go_backwards( *

time_major( :\ X
4
_output_shapes"
 :                  
 
_user_specified_nameinputs:PL
(
_output_shapes
:         а
 
_user_specified_nameinit_h:PL
(
_output_shapes
:         а
 
_user_specified_nameinit_c:GC

_output_shapes
:	ђ	
 
_user_specified_namekernel:RN
 
_output_shapes
:
ађ	
*
_user_specified_namerecurrent_kernel:A=

_output_shapes	
:ђ	

_user_specified_namebias
№)
л
while_body_1208238
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_matmul_kernel_0%
!while_matmul_1_recurrent_kernel_0
while_biasadd_bias_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_matmul_kernel#
while_matmul_1_recurrent_kernel
while_biasadd_biasѕ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       д
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0њ
while/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while_matmul_kernel_0*
T0*(
_output_shapes
:         ђ	Ѓ
while/MatMul_1MatMulwhile_placeholder_2!while_matmul_1_recurrent_kernel_0*
T0*(
_output_shapes
:         ђ	w
	while/addAddV2while/MatMul:product:0while/MatMul_1:product:0*
T0*(
_output_shapes
:         ђ	p
while/BiasAddBiasAddwhile/add:z:0while_biasadd_bias_0*
T0*(
_output_shapes
:         ђ	W
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :╠
while/splitSplitwhile/split/split_dim:output:0while/BiasAdd:output:0*
T0*d
_output_shapesR
P:         а:         а:         а:         а*
	num_splita
while/SigmoidSigmoidwhile/split:output:0*
T0*(
_output_shapes
:         аc
while/Sigmoid_1Sigmoidwhile/split:output:1*
T0*(
_output_shapes
:         аm
	while/mulMulwhile/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:         а[

while/TanhTanhwhile/split:output:2*
T0*(
_output_shapes
:         аh
while/mul_1Mulwhile/Sigmoid:y:0while/Tanh:y:0*
T0*(
_output_shapes
:         аg
while/add_1AddV2while/mul:z:0while/mul_1:z:0*
T0*(
_output_shapes
:         аc
while/Sigmoid_2Sigmoidwhile/split:output:3*
T0*(
_output_shapes
:         аX
while/Tanh_1Tanhwhile/add_1:z:0*
T0*(
_output_shapes
:         аl
while/mul_2Mulwhile/Sigmoid_2:y:0while/Tanh_1:y:0*
T0*(
_output_shapes
:         аr
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : Я
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/mul_2:z:0*
_output_shapes
: *
element_dtype0:жУмO
while/add_2/yConst*
_output_shapes
: *
dtype0*
value	B :`
while/add_2AddV2while_placeholderwhile/add_2/y:output:0*
T0*
_output_shapes
: O
while/add_3/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_3AddV2while_while_loop_counterwhile/add_3/y:output:0*
T0*
_output_shapes
: L
while/IdentityIdentitywhile/add_3:z:0*
T0*
_output_shapes
: ]
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: N
while/Identity_2Identitywhile/add_2:z:0*
T0*
_output_shapes
: y
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: `
while/Identity_4Identitywhile/mul_2:z:0*
T0*(
_output_shapes
:         а`
while/Identity_5Identitywhile/add_1:z:0*
T0*(
_output_shapes
:         а"*
while_biasadd_biaswhile_biasadd_bias_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"D
while_matmul_1_recurrent_kernel!while_matmul_1_recurrent_kernel_0",
while_matmul_kernelwhile_matmul_kernel_0",
while_strided_slicewhile_strided_slice_0"е
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*e
_input_shapesT
R: : : : :         а:         а: : :	ђ	:
ађ	:ђ	: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:         а:.*
(
_output_shapes
:         а:

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	ђ	:&	"
 
_output_shapes
:
ађ	:!


_output_shapes	
:ђ	
╚
▄
J__inference_sequential_13_layer_call_and_return_conditional_losses_1210043

inputs"
lstm_13_1210029:	ђ	#
lstm_13_1210031:
ађ	
lstm_13_1210033:	ђ	#
dense_13_1210036:	а
dense_13_1210038:
identityѕб dense_13/StatefulPartitionedCallбlstm_13/StatefulPartitionedCallЃ
lstm_13/StatefulPartitionedCallStatefulPartitionedCallinputslstm_13_1210029lstm_13_1210031lstm_13_1210033*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         а*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_lstm_13_layer_call_and_return_conditional_losses_1210000Ћ
 dense_13/StatefulPartitionedCallStatefulPartitionedCall(lstm_13/StatefulPartitionedCall:output:0dense_13_1210036dense_13_1210038*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *N
fIRG
E__inference_dense_13_layer_call_and_return_conditional_losses_1209505С
reshape_13/PartitionedCallPartitionedCall)dense_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *P
fKRI
G__inference_reshape_13_layer_call_and_return_conditional_losses_1209524v
IdentityIdentity#reshape_13/PartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         І
NoOpNoOp!^dense_13/StatefulPartitionedCall ^lstm_13/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:         : : : : : 2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall2B
lstm_13/StatefulPartitionedCalllstm_13/StatefulPartitionedCall:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
џ	
к
while_cond_1208237
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice5
1while_while_cond_1208237___redundant_placeholder05
1while_while_cond_1208237___redundant_placeholder15
1while_while_cond_1208237___redundant_placeholder25
1while_while_cond_1208237___redundant_placeholder3
while_identity
`

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :         а:         а: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:         а:.*
(
_output_shapes
:         а:

_output_shapes
: :

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
Ж
э
1__inference_change_layer_12_layer_call_fn_1210199
input_1
unknown:	ђ	
	unknown_0:
ађ	
	unknown_1:	ђ	
	unknown_2:	а
	unknown_3:
identityѕбStatefulPartitionedCallЇ
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         *'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8ѓ *U
fPRN
L__inference_change_layer_12_layer_call_and_return_conditional_losses_1210171s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:         : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:         
!
_user_specified_name	input_1
№)
л
while_body_1212210
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_matmul_kernel_0%
!while_matmul_1_recurrent_kernel_0
while_biasadd_bias_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_matmul_kernel#
while_matmul_1_recurrent_kernel
while_biasadd_biasѕ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       д
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0њ
while/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while_matmul_kernel_0*
T0*(
_output_shapes
:         ђ	Ѓ
while/MatMul_1MatMulwhile_placeholder_2!while_matmul_1_recurrent_kernel_0*
T0*(
_output_shapes
:         ђ	w
	while/addAddV2while/MatMul:product:0while/MatMul_1:product:0*
T0*(
_output_shapes
:         ђ	p
while/BiasAddBiasAddwhile/add:z:0while_biasadd_bias_0*
T0*(
_output_shapes
:         ђ	W
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :╠
while/splitSplitwhile/split/split_dim:output:0while/BiasAdd:output:0*
T0*d
_output_shapesR
P:         а:         а:         а:         а*
	num_splita
while/SigmoidSigmoidwhile/split:output:0*
T0*(
_output_shapes
:         аc
while/Sigmoid_1Sigmoidwhile/split:output:1*
T0*(
_output_shapes
:         аm
	while/mulMulwhile/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:         а[

while/TanhTanhwhile/split:output:2*
T0*(
_output_shapes
:         аh
while/mul_1Mulwhile/Sigmoid:y:0while/Tanh:y:0*
T0*(
_output_shapes
:         аg
while/add_1AddV2while/mul:z:0while/mul_1:z:0*
T0*(
_output_shapes
:         аc
while/Sigmoid_2Sigmoidwhile/split:output:3*
T0*(
_output_shapes
:         аX
while/Tanh_1Tanhwhile/add_1:z:0*
T0*(
_output_shapes
:         аl
while/mul_2Mulwhile/Sigmoid_2:y:0while/Tanh_1:y:0*
T0*(
_output_shapes
:         аr
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : Я
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/mul_2:z:0*
_output_shapes
: *
element_dtype0:жУмO
while/add_2/yConst*
_output_shapes
: *
dtype0*
value	B :`
while/add_2AddV2while_placeholderwhile/add_2/y:output:0*
T0*
_output_shapes
: O
while/add_3/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_3AddV2while_while_loop_counterwhile/add_3/y:output:0*
T0*
_output_shapes
: L
while/IdentityIdentitywhile/add_3:z:0*
T0*
_output_shapes
: ]
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: N
while/Identity_2Identitywhile/add_2:z:0*
T0*
_output_shapes
: y
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: `
while/Identity_4Identitywhile/mul_2:z:0*
T0*(
_output_shapes
:         а`
while/Identity_5Identitywhile/add_1:z:0*
T0*(
_output_shapes
:         а"*
while_biasadd_biaswhile_biasadd_bias_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"D
while_matmul_1_recurrent_kernel!while_matmul_1_recurrent_kernel_0",
while_matmul_kernelwhile_matmul_kernel_0",
while_strided_slicewhile_strided_slice_0"е
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*e
_input_shapesT
R: : : : :         а:         а: : :	ђ	:
ађ	:ђ	: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:         а:.*
(
_output_shapes
:         а:

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	ђ	:&	"
 
_output_shapes
:
ађ	:!


_output_shapes	
:ђ	
ћ
╣
)__inference_lstm_13_layer_call_fn_1212118
inputs_0
unknown:	ђ	
	unknown_0:
ађ	
	unknown_1:	ђ	
identityѕбStatefulPartitionedCallж
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         а*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_lstm_13_layer_call_and_return_conditional_losses_1209041p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         а`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :                  
"
_user_specified_name
inputs_0
тB
═
*__inference_gpu_lstm_with_fallback_1212392

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4ѕc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :                  P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : p

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0*,
_output_shapes
:         аR
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : t
ExpandDims_1
ExpandDimsinit_cExpandDims_1/dim:output:0*
T0*,
_output_shapes
:         аQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ї
splitSplitsplit/split_dim:output:0kernel*
T0*@
_output_shapes.
,:	а:	а:	а:	а*
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ъ
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*D
_output_shapes2
0:
аа:
аа:
аа:
аа*
	num_splite
zeros_like/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB:ђ	U
zeros_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    x

zeros_likeFill#zeros_like/shape_as_tensor:output:0zeros_like/Const:output:0*
T0*
_output_shapes	
:ђ	M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : r
concatConcatV2zeros_like:output:0biasconcat/axis:output:0*
N*
T0*
_output_shapes	
:ђS
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : Ц
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*L
_output_shapes:
8:а:а:а:а:а:а:а:а*
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
         a
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       m
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes
:	аY
ReshapeReshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:аa
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       m
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes
:	а[
	Reshape_1Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:аa
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       m
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes
:	а[
	Reshape_2Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:аa
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       m
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0*
_output_shapes
:	а[
	Reshape_3Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes	
:аa
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       p
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0* 
_output_shapes
:
аа\
	Reshape_4Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes

:ђѕa
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       p
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0* 
_output_shapes
:
аа\
	Reshape_5Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes

:ђѕa
transpose_7/permConst*
_output_shapes
:*
dtype0*
valueB"       p
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0* 
_output_shapes
:
аа\
	Reshape_6Reshapetranspose_7:y:0Const:output:0*
T0*
_output_shapes

:ђѕa
transpose_8/permConst*
_output_shapes
:*
dtype0*
valueB"       p
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0* 
_output_shapes
:
аа\
	Reshape_7Reshapetranspose_8:y:0Const:output:0*
T0*
_output_shapes

:ђѕ\
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes	
:а\
	Reshape_9Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes	
:а]

Reshape_10Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes	
:а]

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes	
:а]

Reshape_12Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes	
:а]

Reshape_13Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes	
:а]

Reshape_14Reshapesplit_2:output:6Const:output:0*
T0*
_output_shapes	
:а]

Reshape_15Reshapesplit_2:output:7Const:output:0*
T0*
_output_shapes	
:аO
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : а
concat_1ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T0*
_output_shapes

:ђ╗о
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1:output:0*
T0*i
_output_shapesW
U:                  а:         а:         а:f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
         _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Т
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         а*
shrink_axis_maskq
SqueezeSqueezeCudnnRNN:output_h:0*
T0*(
_output_shapes
:         а*
squeeze_dims
 s
	Squeeze_1SqueezeCudnnRNN:output_c:0*
T0*(
_output_shapes
:         а*
squeeze_dims
 R
ExpandDims_2/dimConst*
_output_shapes
: *
dtype0*
value	B :ё
ExpandDims_2
ExpandDimsstrided_slice:output:0ExpandDims_2/dim:output:0*
T0*,
_output_shapes
:         а[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @_
IdentityIdentitystrided_slice:output:0*
T0*(
_output_shapes
:         аd

Identity_1IdentityExpandDims_2:output:0*
T0*,
_output_shapes
:         а[

Identity_2IdentitySqueeze:output:0*
T0*(
_output_shapes
:         а]

Identity_3IdentitySqueeze_1:output:0*
T0*(
_output_shapes
:         аI

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*y
_input_shapesh
f:                  :         а:         а:	ђ	:
ађ	:ђ	*=
api_implements+)lstm_8e4ab49d-552b-4e19-8094-3b0af108a7d4*
api_preferred_deviceGPU*
go_backwards( *

time_major( :\ X
4
_output_shapes"
 :                  
 
_user_specified_nameinputs:PL
(
_output_shapes
:         а
 
_user_specified_nameinit_h:PL
(
_output_shapes
:         а
 
_user_specified_nameinit_c:GC

_output_shapes
:	ђ	
 
_user_specified_namekernel:RN
 
_output_shapes
:
ађ	
*
_user_specified_namerecurrent_kernel:A=

_output_shapes	
:ђ	

_user_specified_namebias
тB
═
*__inference_gpu_lstm_with_fallback_1208862

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4ѕc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :                  P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : p

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0*,
_output_shapes
:         аR
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : t
ExpandDims_1
ExpandDimsinit_cExpandDims_1/dim:output:0*
T0*,
_output_shapes
:         аQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ї
splitSplitsplit/split_dim:output:0kernel*
T0*@
_output_shapes.
,:	а:	а:	а:	а*
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ъ
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*D
_output_shapes2
0:
аа:
аа:
аа:
аа*
	num_splite
zeros_like/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB:ђ	U
zeros_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    x

zeros_likeFill#zeros_like/shape_as_tensor:output:0zeros_like/Const:output:0*
T0*
_output_shapes	
:ђ	M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : r
concatConcatV2zeros_like:output:0biasconcat/axis:output:0*
N*
T0*
_output_shapes	
:ђS
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : Ц
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*L
_output_shapes:
8:а:а:а:а:а:а:а:а*
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
         a
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       m
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes
:	аY
ReshapeReshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:аa
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       m
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes
:	а[
	Reshape_1Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:аa
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       m
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes
:	а[
	Reshape_2Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:аa
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       m
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0*
_output_shapes
:	а[
	Reshape_3Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes	
:аa
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       p
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0* 
_output_shapes
:
аа\
	Reshape_4Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes

:ђѕa
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       p
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0* 
_output_shapes
:
аа\
	Reshape_5Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes

:ђѕa
transpose_7/permConst*
_output_shapes
:*
dtype0*
valueB"       p
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0* 
_output_shapes
:
аа\
	Reshape_6Reshapetranspose_7:y:0Const:output:0*
T0*
_output_shapes

:ђѕa
transpose_8/permConst*
_output_shapes
:*
dtype0*
valueB"       p
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0* 
_output_shapes
:
аа\
	Reshape_7Reshapetranspose_8:y:0Const:output:0*
T0*
_output_shapes

:ђѕ\
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes	
:а\
	Reshape_9Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes	
:а]

Reshape_10Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes	
:а]

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes	
:а]

Reshape_12Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes	
:а]

Reshape_13Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes	
:а]

Reshape_14Reshapesplit_2:output:6Const:output:0*
T0*
_output_shapes	
:а]

Reshape_15Reshapesplit_2:output:7Const:output:0*
T0*
_output_shapes	
:аO
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : а
concat_1ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T0*
_output_shapes

:ђ╗о
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1:output:0*
T0*i
_output_shapesW
U:                  а:         а:         а:f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
         _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Т
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         а*
shrink_axis_maskq
SqueezeSqueezeCudnnRNN:output_h:0*
T0*(
_output_shapes
:         а*
squeeze_dims
 s
	Squeeze_1SqueezeCudnnRNN:output_c:0*
T0*(
_output_shapes
:         а*
squeeze_dims
 R
ExpandDims_2/dimConst*
_output_shapes
: *
dtype0*
value	B :ё
ExpandDims_2
ExpandDimsstrided_slice:output:0ExpandDims_2/dim:output:0*
T0*,
_output_shapes
:         а[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @_
IdentityIdentitystrided_slice:output:0*
T0*(
_output_shapes
:         аd

Identity_1IdentityExpandDims_2:output:0*
T0*,
_output_shapes
:         а[

Identity_2IdentitySqueeze:output:0*
T0*(
_output_shapes
:         а]

Identity_3IdentitySqueeze_1:output:0*
T0*(
_output_shapes
:         аI

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*y
_input_shapesh
f:                  :         а:         а:	ђ	:
ађ	:ђ	*=
api_implements+)lstm_8b39bb2f-e1e5-4618-9947-ffe4540afe4b*
api_preferred_deviceGPU*
go_backwards( *

time_major( :\ X
4
_output_shapes"
 :                  
 
_user_specified_nameinputs:PL
(
_output_shapes
:         а
 
_user_specified_nameinit_h:PL
(
_output_shapes
:         а
 
_user_specified_nameinit_c:GC

_output_shapes
:	ђ	
 
_user_specified_namekernel:RN
 
_output_shapes
:
ађ	
*
_user_specified_namerecurrent_kernel:A=

_output_shapes	
:ђ	

_user_specified_namebias
Б
┐
D__inference_lstm_13_layer_call_and_return_conditional_losses_1210000

inputs/
read_readvariableop_resource:	ђ	2
read_1_readvariableop_resource:
ађ	-
read_2_readvariableop_resource:	ђ	

identity_3ѕбRead/ReadVariableOpбRead_1/ReadVariableOpбRead_2/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Л
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :аs
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:         аS
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :аw
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    s
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:         аq
Read/ReadVariableOpReadVariableOpread_readvariableop_resource*
_output_shapes
:	ђ	*
dtype0[
IdentityIdentityRead/ReadVariableOp:value:0*
T0*
_output_shapes
:	ђ	v
Read_1/ReadVariableOpReadVariableOpread_1_readvariableop_resource* 
_output_shapes
:
ађ	*
dtype0`

Identity_1IdentityRead_1/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ађ	q
Read_2/ReadVariableOpReadVariableOpread_2_readvariableop_resource*
_output_shapes	
:ђ	*
dtype0[

Identity_2IdentityRead_2/ReadVariableOp:value:0*
T0*
_output_shapes	
:ђ	и
PartitionedCallPartitionedCallinputszeros:output:0zeros_1:output:0Identity:output:0Identity_1:output:0Identity_2:output:0*
Tin

2*
Tout	
2*
_collective_manager_ids
 *j
_output_shapesX
V:         а:         а:         а:         а: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ **
f%R#
!__inference_standard_lstm_1209725j

Identity_3IdentityPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         аї
NoOpNoOp^Read/ReadVariableOp^Read_1/ReadVariableOp^Read_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         : : : 2*
Read/ReadVariableOpRead/ReadVariableOp2.
Read_1/ReadVariableOpRead_1/ReadVariableOp2.
Read_2/ReadVariableOpRead_2/ReadVariableOp:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
Ы┬
у
=__inference___backward_gpu_lstm_with_fallback_1208421_1208597
placeholder
placeholder_1
placeholder_2
placeholder_3
placeholder_43
/gradients_expanddims_2_grad_shape_strided_slice)
%gradients_squeeze_grad_shape_cudnnrnn+
'gradients_squeeze_1_grad_shape_cudnnrnn/
+gradients_strided_slice_grad_shape_cudnnrnn!
gradients_zeros_like_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9
5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_15
1gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h,
(gradients_expanddims_1_grad_shape_init_c-
)gradients_concat_1_grad_mod_concat_1_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_permA
=gradients_transpose_7_grad_invertpermutation_transpose_7_permA
=gradients_transpose_8_grad_invertpermutation_transpose_8_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim)
%gradients_concat_grad_mod_concat_axis
identity

identity_1

identity_2

identity_3

identity_4

identity_5ѕ_
gradients/grad_ys_0Identityplaceholder*
T0*(
_output_shapes
:         аe
gradients/grad_ys_1Identityplaceholder_1*
T0*,
_output_shapes
:         аa
gradients/grad_ys_2Identityplaceholder_2*
T0*(
_output_shapes
:         аa
gradients/grad_ys_3Identityplaceholder_3*
T0*(
_output_shapes
:         аO
gradients/grad_ys_4Identityplaceholder_4*
T0*
_output_shapes
: ђ
!gradients/ExpandDims_2_grad/ShapeShape/gradients_expanddims_2_grad_shape_strided_slice*
T0*
_output_shapes
:Ф
#gradients/ExpandDims_2_grad/ReshapeReshapegradients/grad_ys_1:output:0*gradients/ExpandDims_2_grad/Shape:output:0*
T0*(
_output_shapes
:         аq
gradients/Squeeze_grad/ShapeShape%gradients_squeeze_grad_shape_cudnnrnn*
T0*
_output_shapes
:Ц
gradients/Squeeze_grad/ReshapeReshapegradients/grad_ys_2:output:0%gradients/Squeeze_grad/Shape:output:0*
T0*,
_output_shapes
:         аu
gradients/Squeeze_1_grad/ShapeShape'gradients_squeeze_1_grad_shape_cudnnrnn*
T0*
_output_shapes
:Е
 gradients/Squeeze_1_grad/ReshapeReshapegradients/grad_ys_3:output:0'gradients/Squeeze_1_grad/Shape:output:0*
T0*,
_output_shapes
:         ак
gradients/AddNAddNgradients/grad_ys_0:output:0,gradients/ExpandDims_2_grad/Reshape:output:0*
N*
T0*&
_class
loc:@gradients/grad_ys_0*(
_output_shapes
:         а}
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
:є
3gradients/strided_slice_grad/StridedSliceGrad/beginConst*
_output_shapes
:*
dtype0*
valueB:
         {
1gradients/strided_slice_grad/StridedSliceGrad/endConst*
_output_shapes
:*
dtype0*
valueB: 
5gradients/strided_slice_grad/StridedSliceGrad/stridesConst*
_output_shapes
:*
dtype0*
valueB:Б
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0<gradients/strided_slice_grad/StridedSliceGrad/begin:output:0:gradients/strided_slice_grad/StridedSliceGrad/end:output:0>gradients/strided_slice_grad/StridedSliceGrad/strides:output:0gradients/AddN:sum:0*
Index0*
T0*5
_output_shapes#
!:                  а*
shrink_axis_maskc
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
:«
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_11gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1+gradients_strided_slice_grad_shape_cudnnrnn%gradients_squeeze_grad_shape_cudnnrnn'gradients_squeeze_1_grad_shape_cudnnrnn6gradients/strided_slice_grad/StridedSliceGrad:output:0'gradients/Squeeze_grad/Reshape:output:0)gradients/Squeeze_1_grad/Reshape:output:0gradients_zeros_like_cudnnrnn*
T0*l
_output_shapesZ
X:                  :         а:         а:ђ╗ќ
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:┘
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*4
_output_shapes"
 :                  u
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
:к
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*(
_output_shapes
:         аy
!gradients/ExpandDims_1_grad/ShapeShape(gradients_expanddims_1_grad_shape_init_c*
T0*
_output_shapes
:╩
#gradients/ExpandDims_1_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_c_backprop:0*gradients/ExpandDims_1_grad/Shape:output:0*
T0*(
_output_shapes
:         а^
gradients/concat_1_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :џ
gradients/concat_1_grad/modFloorMod)gradients_concat_1_grad_mod_concat_1_axis%gradients/concat_1_grad/Rank:output:0*
T0*
_output_shapes
: h
gradients/concat_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:аj
gradients/concat_1_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:аj
gradients/concat_1_grad/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:аj
gradients/concat_1_grad/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:аk
gradients/concat_1_grad/Shape_4Const*
_output_shapes
:*
dtype0*
valueB:ђѕk
gradients/concat_1_grad/Shape_5Const*
_output_shapes
:*
dtype0*
valueB:ђѕk
gradients/concat_1_grad/Shape_6Const*
_output_shapes
:*
dtype0*
valueB:ђѕk
gradients/concat_1_grad/Shape_7Const*
_output_shapes
:*
dtype0*
valueB:ђѕj
gradients/concat_1_grad/Shape_8Const*
_output_shapes
:*
dtype0*
valueB:аj
gradients/concat_1_grad/Shape_9Const*
_output_shapes
:*
dtype0*
valueB:аk
 gradients/concat_1_grad/Shape_10Const*
_output_shapes
:*
dtype0*
valueB:аk
 gradients/concat_1_grad/Shape_11Const*
_output_shapes
:*
dtype0*
valueB:аk
 gradients/concat_1_grad/Shape_12Const*
_output_shapes
:*
dtype0*
valueB:аk
 gradients/concat_1_grad/Shape_13Const*
_output_shapes
:*
dtype0*
valueB:аk
 gradients/concat_1_grad/Shape_14Const*
_output_shapes
:*
dtype0*
valueB:аk
 gradients/concat_1_grad/Shape_15Const*
_output_shapes
:*
dtype0*
valueB:аЭ
$gradients/concat_1_grad/ConcatOffsetConcatOffsetgradients/concat_1_grad/mod:z:0&gradients/concat_1_grad/Shape:output:0(gradients/concat_1_grad/Shape_1:output:0(gradients/concat_1_grad/Shape_2:output:0(gradients/concat_1_grad/Shape_3:output:0(gradients/concat_1_grad/Shape_4:output:0(gradients/concat_1_grad/Shape_5:output:0(gradients/concat_1_grad/Shape_6:output:0(gradients/concat_1_grad/Shape_7:output:0(gradients/concat_1_grad/Shape_8:output:0(gradients/concat_1_grad/Shape_9:output:0)gradients/concat_1_grad/Shape_10:output:0)gradients/concat_1_grad/Shape_11:output:0)gradients/concat_1_grad/Shape_12:output:0)gradients/concat_1_grad/Shape_13:output:0)gradients/concat_1_grad/Shape_14:output:0)gradients/concat_1_grad/Shape_15:output:0*
N*t
_output_shapesb
`::::::::::::::::В
gradients/concat_1_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:0&gradients/concat_1_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:а­
gradients/concat_1_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:1(gradients/concat_1_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:а­
gradients/concat_1_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:2(gradients/concat_1_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes	
:а­
gradients/concat_1_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:3(gradients/concat_1_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes	
:аы
gradients/concat_1_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:4(gradients/concat_1_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes

:ђѕы
gradients/concat_1_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:5(gradients/concat_1_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes

:ђѕы
gradients/concat_1_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:6(gradients/concat_1_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes

:ђѕы
gradients/concat_1_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:7(gradients/concat_1_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes

:ђѕ­
gradients/concat_1_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:8(gradients/concat_1_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes	
:а­
gradients/concat_1_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:9(gradients/concat_1_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes	
:аз
 gradients/concat_1_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:10)gradients/concat_1_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes	
:аз
 gradients/concat_1_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:11)gradients/concat_1_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes	
:аз
 gradients/concat_1_grad/Slice_12Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:12)gradients/concat_1_grad/Shape_12:output:0*
Index0*
T0*
_output_shapes	
:аз
 gradients/concat_1_grad/Slice_13Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:13)gradients/concat_1_grad/Shape_13:output:0*
Index0*
T0*
_output_shapes	
:аз
 gradients/concat_1_grad/Slice_14Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:14)gradients/concat_1_grad/Shape_14:output:0*
Index0*
T0*
_output_shapes	
:аз
 gradients/concat_1_grad/Slice_15Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:15)gradients/concat_1_grad/Shape_15:output:0*
Index0*
T0*
_output_shapes	
:аm
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      б
gradients/Reshape_grad/ReshapeReshape&gradients/concat_1_grad/Slice:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes
:	аo
gradients/Reshape_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      е
 gradients/Reshape_1_grad/ReshapeReshape(gradients/concat_1_grad/Slice_1:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0*
_output_shapes
:	аo
gradients/Reshape_2_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      е
 gradients/Reshape_2_grad/ReshapeReshape(gradients/concat_1_grad/Slice_2:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0*
_output_shapes
:	аo
gradients/Reshape_3_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      е
 gradients/Reshape_3_grad/ReshapeReshape(gradients/concat_1_grad/Slice_3:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0*
_output_shapes
:	аo
gradients/Reshape_4_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      Е
 gradients/Reshape_4_grad/ReshapeReshape(gradients/concat_1_grad/Slice_4:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0* 
_output_shapes
:
ааo
gradients/Reshape_5_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      Е
 gradients/Reshape_5_grad/ReshapeReshape(gradients/concat_1_grad/Slice_5:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0* 
_output_shapes
:
ааo
gradients/Reshape_6_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      Е
 gradients/Reshape_6_grad/ReshapeReshape(gradients/concat_1_grad/Slice_6:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0* 
_output_shapes
:
ааo
gradients/Reshape_7_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      Е
 gradients/Reshape_7_grad/ReshapeReshape(gradients/concat_1_grad/Slice_7:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0* 
_output_shapes
:
ааi
gradients/Reshape_8_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:ац
 gradients/Reshape_8_grad/ReshapeReshape(gradients/concat_1_grad/Slice_8:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes	
:аi
gradients/Reshape_9_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:ац
 gradients/Reshape_9_grad/ReshapeReshape(gradients/concat_1_grad/Slice_9:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes	
:аj
gradients/Reshape_10_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:аД
!gradients/Reshape_10_grad/ReshapeReshape)gradients/concat_1_grad/Slice_10:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes	
:аj
gradients/Reshape_11_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:аД
!gradients/Reshape_11_grad/ReshapeReshape)gradients/concat_1_grad/Slice_11:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes	
:аj
gradients/Reshape_12_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:аД
!gradients/Reshape_12_grad/ReshapeReshape)gradients/concat_1_grad/Slice_12:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes	
:аj
gradients/Reshape_13_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:аД
!gradients/Reshape_13_grad/ReshapeReshape)gradients/concat_1_grad/Slice_13:output:0(gradients/Reshape_13_grad/Shape:output:0*
T0*
_output_shapes	
:аj
gradients/Reshape_14_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:аД
!gradients/Reshape_14_grad/ReshapeReshape)gradients/concat_1_grad/Slice_14:output:0(gradients/Reshape_14_grad/Shape:output:0*
T0*
_output_shapes	
:аj
gradients/Reshape_15_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:аД
!gradients/Reshape_15_grad/ReshapeReshape)gradients/concat_1_grad/Slice_15:output:0(gradients/Reshape_15_grad/Shape:output:0*
T0*
_output_shapes	
:аю
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:Х
$gradients/transpose_1_grad/transpose	Transpose'gradients/Reshape_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0*
_output_shapes
:	аю
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:И
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0*
_output_shapes
:	аю
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:И
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0*
_output_shapes
:	аю
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:И
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0*
_output_shapes
:	аю
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:╣
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
ааю
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:╣
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
ааю
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:╣
$gradients/transpose_7_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
ааю
,gradients/transpose_8_grad/InvertPermutationInvertPermutation=gradients_transpose_8_grad_invertpermutation_transpose_8_perm*
_output_shapes
:╣
$gradients/transpose_8_grad/transpose	Transpose)gradients/Reshape_7_grad/Reshape:output:00gradients/transpose_8_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
аау
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0*gradients/Reshape_13_grad/Reshape:output:0*gradients/Reshape_14_grad/Reshape:output:0*gradients/Reshape_15_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
N*
T0*
_output_shapes	
:ђ»
gradients/split_grad/concatConcatV2(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
N*
T0*
_output_shapes
:	ђ	Х
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0(gradients/transpose_7_grad/transpose:y:0(gradients/transpose_8_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
N*
T0* 
_output_shapes
:
ађ	\
gradients/concat_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :њ
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: f
gradients/concat_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:ђ	h
gradients/concat_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:ђ	╩
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0*
N* 
_output_shapes
::м
gradients/concat_grad/SliceSlice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:ђ	о
gradients/concat_grad/Slice_1Slice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:ђ	{
IdentityIdentity&gradients/transpose_grad/transpose:y:0*
T0*4
_output_shapes"
 :                  u

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0*
T0*(
_output_shapes
:         аw

Identity_2Identity,gradients/ExpandDims_1_grad/Reshape:output:0*
T0*(
_output_shapes
:         аf

Identity_3Identity$gradients/split_grad/concat:output:0*
T0*
_output_shapes
:	ђ	i

Identity_4Identity&gradients/split_1_grad/concat:output:0*
T0* 
_output_shapes
:
ађ	d

Identity_5Identity&gradients/concat_grad/Slice_1:output:0*
T0*
_output_shapes	
:ђ	"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*(
_construction_contextkEagerRuntime*ћ
_input_shapesѓ
 :         а:         а:         а:         а: :         а:         а:         а:                  а::                  :         а:         а:ђ╗::         а:         а: ::::::::: : : : *=
api_implements+)lstm_d272952e-0e4d-4b32-9764-a4b66b442161*
api_preferred_deviceGPU*C
forward_function_name*(__forward_gpu_lstm_with_fallback_1208596*
go_backwards( *

time_major( :. *
(
_output_shapes
:         а:2.
,
_output_shapes
:         а:.*
(
_output_shapes
:         а:.*
(
_output_shapes
:         а:

_output_shapes
: :.*
(
_output_shapes
:         а:2.
,
_output_shapes
:         а:2.
,
_output_shapes
:         а:;7
5
_output_shapes#
!:                  а:	

_output_shapes
:::
6
4
_output_shapes"
 :                  :2.
,
_output_shapes
:         а:2.
,
_output_shapes
:         а:"

_output_shapes

:ђ╗: 

_output_shapes
::.*
(
_output_shapes
:         а:.*
(
_output_shapes
:         а:

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Ц
П
L__inference_change_layer_12_layer_call_and_return_conditional_losses_1210125

inputs(
sequential_13_1210112:	ђ	)
sequential_13_1210114:
ађ	$
sequential_13_1210116:	ђ	(
sequential_13_1210118:	а#
sequential_13_1210120:
identityѕб%sequential_13/StatefulPartitionedCallо
%sequential_13/StatefulPartitionedCallStatefulPartitionedCallinputssequential_13_1210112sequential_13_1210114sequential_13_1210116sequential_13_1210118sequential_13_1210120*
Tin

2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         *'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8ѓ *S
fNRL
J__inference_sequential_13_layer_call_and_return_conditional_losses_1209527z
addAddV2inputs.sequential_13/StatefulPartitionedCall:output:0*
T0*+
_output_shapes
:         Z
IdentityIdentityadd:z:0^NoOp*
T0*+
_output_shapes
:         n
NoOpNoOp&^sequential_13/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:         : : : : : 2N
%sequential_13/StatefulPartitionedCall%sequential_13/StatefulPartitionedCall:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
ф┬
у
=__inference___backward_gpu_lstm_with_fallback_1207970_1208146
placeholder
placeholder_1
placeholder_2
placeholder_3
placeholder_43
/gradients_expanddims_2_grad_shape_strided_slice)
%gradients_squeeze_grad_shape_cudnnrnn+
'gradients_squeeze_1_grad_shape_cudnnrnn/
+gradients_strided_slice_grad_shape_cudnnrnn!
gradients_zeros_like_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9
5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_15
1gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h,
(gradients_expanddims_1_grad_shape_init_c-
)gradients_concat_1_grad_mod_concat_1_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_permA
=gradients_transpose_7_grad_invertpermutation_transpose_7_permA
=gradients_transpose_8_grad_invertpermutation_transpose_8_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim)
%gradients_concat_grad_mod_concat_axis
identity

identity_1

identity_2

identity_3

identity_4

identity_5ѕ_
gradients/grad_ys_0Identityplaceholder*
T0*(
_output_shapes
:         аe
gradients/grad_ys_1Identityplaceholder_1*
T0*,
_output_shapes
:         аa
gradients/grad_ys_2Identityplaceholder_2*
T0*(
_output_shapes
:         аa
gradients/grad_ys_3Identityplaceholder_3*
T0*(
_output_shapes
:         аO
gradients/grad_ys_4Identityplaceholder_4*
T0*
_output_shapes
: ђ
!gradients/ExpandDims_2_grad/ShapeShape/gradients_expanddims_2_grad_shape_strided_slice*
T0*
_output_shapes
:Ф
#gradients/ExpandDims_2_grad/ReshapeReshapegradients/grad_ys_1:output:0*gradients/ExpandDims_2_grad/Shape:output:0*
T0*(
_output_shapes
:         аq
gradients/Squeeze_grad/ShapeShape%gradients_squeeze_grad_shape_cudnnrnn*
T0*
_output_shapes
:Ц
gradients/Squeeze_grad/ReshapeReshapegradients/grad_ys_2:output:0%gradients/Squeeze_grad/Shape:output:0*
T0*,
_output_shapes
:         аu
gradients/Squeeze_1_grad/ShapeShape'gradients_squeeze_1_grad_shape_cudnnrnn*
T0*
_output_shapes
:Е
 gradients/Squeeze_1_grad/ReshapeReshapegradients/grad_ys_3:output:0'gradients/Squeeze_1_grad/Shape:output:0*
T0*,
_output_shapes
:         ак
gradients/AddNAddNgradients/grad_ys_0:output:0,gradients/ExpandDims_2_grad/Reshape:output:0*
N*
T0*&
_class
loc:@gradients/grad_ys_0*(
_output_shapes
:         а}
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
:є
3gradients/strided_slice_grad/StridedSliceGrad/beginConst*
_output_shapes
:*
dtype0*
valueB:
         {
1gradients/strided_slice_grad/StridedSliceGrad/endConst*
_output_shapes
:*
dtype0*
valueB: 
5gradients/strided_slice_grad/StridedSliceGrad/stridesConst*
_output_shapes
:*
dtype0*
valueB:џ
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0<gradients/strided_slice_grad/StridedSliceGrad/begin:output:0:gradients/strided_slice_grad/StridedSliceGrad/end:output:0>gradients/strided_slice_grad/StridedSliceGrad/strides:output:0gradients/AddN:sum:0*
Index0*
T0*,
_output_shapes
:         а*
shrink_axis_maskc
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
:Ц
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_11gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1+gradients_strided_slice_grad_shape_cudnnrnn%gradients_squeeze_grad_shape_cudnnrnn'gradients_squeeze_1_grad_shape_cudnnrnn6gradients/strided_slice_grad/StridedSliceGrad:output:0'gradients/Squeeze_grad/Reshape:output:0)gradients/Squeeze_1_grad/Reshape:output:0gradients_zeros_like_cudnnrnn*
T0*c
_output_shapesQ
O:         :         а:         а:ђ╗ќ
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:л
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*+
_output_shapes
:         u
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
:к
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*(
_output_shapes
:         аy
!gradients/ExpandDims_1_grad/ShapeShape(gradients_expanddims_1_grad_shape_init_c*
T0*
_output_shapes
:╩
#gradients/ExpandDims_1_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_c_backprop:0*gradients/ExpandDims_1_grad/Shape:output:0*
T0*(
_output_shapes
:         а^
gradients/concat_1_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :џ
gradients/concat_1_grad/modFloorMod)gradients_concat_1_grad_mod_concat_1_axis%gradients/concat_1_grad/Rank:output:0*
T0*
_output_shapes
: h
gradients/concat_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:аj
gradients/concat_1_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:аj
gradients/concat_1_grad/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:аj
gradients/concat_1_grad/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:аk
gradients/concat_1_grad/Shape_4Const*
_output_shapes
:*
dtype0*
valueB:ђѕk
gradients/concat_1_grad/Shape_5Const*
_output_shapes
:*
dtype0*
valueB:ђѕk
gradients/concat_1_grad/Shape_6Const*
_output_shapes
:*
dtype0*
valueB:ђѕk
gradients/concat_1_grad/Shape_7Const*
_output_shapes
:*
dtype0*
valueB:ђѕj
gradients/concat_1_grad/Shape_8Const*
_output_shapes
:*
dtype0*
valueB:аj
gradients/concat_1_grad/Shape_9Const*
_output_shapes
:*
dtype0*
valueB:аk
 gradients/concat_1_grad/Shape_10Const*
_output_shapes
:*
dtype0*
valueB:аk
 gradients/concat_1_grad/Shape_11Const*
_output_shapes
:*
dtype0*
valueB:аk
 gradients/concat_1_grad/Shape_12Const*
_output_shapes
:*
dtype0*
valueB:аk
 gradients/concat_1_grad/Shape_13Const*
_output_shapes
:*
dtype0*
valueB:аk
 gradients/concat_1_grad/Shape_14Const*
_output_shapes
:*
dtype0*
valueB:аk
 gradients/concat_1_grad/Shape_15Const*
_output_shapes
:*
dtype0*
valueB:аЭ
$gradients/concat_1_grad/ConcatOffsetConcatOffsetgradients/concat_1_grad/mod:z:0&gradients/concat_1_grad/Shape:output:0(gradients/concat_1_grad/Shape_1:output:0(gradients/concat_1_grad/Shape_2:output:0(gradients/concat_1_grad/Shape_3:output:0(gradients/concat_1_grad/Shape_4:output:0(gradients/concat_1_grad/Shape_5:output:0(gradients/concat_1_grad/Shape_6:output:0(gradients/concat_1_grad/Shape_7:output:0(gradients/concat_1_grad/Shape_8:output:0(gradients/concat_1_grad/Shape_9:output:0)gradients/concat_1_grad/Shape_10:output:0)gradients/concat_1_grad/Shape_11:output:0)gradients/concat_1_grad/Shape_12:output:0)gradients/concat_1_grad/Shape_13:output:0)gradients/concat_1_grad/Shape_14:output:0)gradients/concat_1_grad/Shape_15:output:0*
N*t
_output_shapesb
`::::::::::::::::В
gradients/concat_1_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:0&gradients/concat_1_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:а­
gradients/concat_1_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:1(gradients/concat_1_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:а­
gradients/concat_1_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:2(gradients/concat_1_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes	
:а­
gradients/concat_1_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:3(gradients/concat_1_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes	
:аы
gradients/concat_1_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:4(gradients/concat_1_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes

:ђѕы
gradients/concat_1_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:5(gradients/concat_1_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes

:ђѕы
gradients/concat_1_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:6(gradients/concat_1_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes

:ђѕы
gradients/concat_1_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:7(gradients/concat_1_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes

:ђѕ­
gradients/concat_1_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:8(gradients/concat_1_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes	
:а­
gradients/concat_1_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:9(gradients/concat_1_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes	
:аз
 gradients/concat_1_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:10)gradients/concat_1_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes	
:аз
 gradients/concat_1_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:11)gradients/concat_1_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes	
:аз
 gradients/concat_1_grad/Slice_12Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:12)gradients/concat_1_grad/Shape_12:output:0*
Index0*
T0*
_output_shapes	
:аз
 gradients/concat_1_grad/Slice_13Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:13)gradients/concat_1_grad/Shape_13:output:0*
Index0*
T0*
_output_shapes	
:аз
 gradients/concat_1_grad/Slice_14Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:14)gradients/concat_1_grad/Shape_14:output:0*
Index0*
T0*
_output_shapes	
:аз
 gradients/concat_1_grad/Slice_15Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:15)gradients/concat_1_grad/Shape_15:output:0*
Index0*
T0*
_output_shapes	
:аm
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      б
gradients/Reshape_grad/ReshapeReshape&gradients/concat_1_grad/Slice:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes
:	аo
gradients/Reshape_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      е
 gradients/Reshape_1_grad/ReshapeReshape(gradients/concat_1_grad/Slice_1:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0*
_output_shapes
:	аo
gradients/Reshape_2_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      е
 gradients/Reshape_2_grad/ReshapeReshape(gradients/concat_1_grad/Slice_2:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0*
_output_shapes
:	аo
gradients/Reshape_3_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      е
 gradients/Reshape_3_grad/ReshapeReshape(gradients/concat_1_grad/Slice_3:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0*
_output_shapes
:	аo
gradients/Reshape_4_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      Е
 gradients/Reshape_4_grad/ReshapeReshape(gradients/concat_1_grad/Slice_4:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0* 
_output_shapes
:
ааo
gradients/Reshape_5_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      Е
 gradients/Reshape_5_grad/ReshapeReshape(gradients/concat_1_grad/Slice_5:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0* 
_output_shapes
:
ааo
gradients/Reshape_6_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      Е
 gradients/Reshape_6_grad/ReshapeReshape(gradients/concat_1_grad/Slice_6:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0* 
_output_shapes
:
ааo
gradients/Reshape_7_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      Е
 gradients/Reshape_7_grad/ReshapeReshape(gradients/concat_1_grad/Slice_7:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0* 
_output_shapes
:
ааi
gradients/Reshape_8_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:ац
 gradients/Reshape_8_grad/ReshapeReshape(gradients/concat_1_grad/Slice_8:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes	
:аi
gradients/Reshape_9_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:ац
 gradients/Reshape_9_grad/ReshapeReshape(gradients/concat_1_grad/Slice_9:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes	
:аj
gradients/Reshape_10_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:аД
!gradients/Reshape_10_grad/ReshapeReshape)gradients/concat_1_grad/Slice_10:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes	
:аj
gradients/Reshape_11_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:аД
!gradients/Reshape_11_grad/ReshapeReshape)gradients/concat_1_grad/Slice_11:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes	
:аj
gradients/Reshape_12_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:аД
!gradients/Reshape_12_grad/ReshapeReshape)gradients/concat_1_grad/Slice_12:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes	
:аj
gradients/Reshape_13_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:аД
!gradients/Reshape_13_grad/ReshapeReshape)gradients/concat_1_grad/Slice_13:output:0(gradients/Reshape_13_grad/Shape:output:0*
T0*
_output_shapes	
:аj
gradients/Reshape_14_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:аД
!gradients/Reshape_14_grad/ReshapeReshape)gradients/concat_1_grad/Slice_14:output:0(gradients/Reshape_14_grad/Shape:output:0*
T0*
_output_shapes	
:аj
gradients/Reshape_15_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:аД
!gradients/Reshape_15_grad/ReshapeReshape)gradients/concat_1_grad/Slice_15:output:0(gradients/Reshape_15_grad/Shape:output:0*
T0*
_output_shapes	
:аю
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:Х
$gradients/transpose_1_grad/transpose	Transpose'gradients/Reshape_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0*
_output_shapes
:	аю
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:И
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0*
_output_shapes
:	аю
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:И
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0*
_output_shapes
:	аю
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:И
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0*
_output_shapes
:	аю
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:╣
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
ааю
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:╣
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
ааю
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:╣
$gradients/transpose_7_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
ааю
,gradients/transpose_8_grad/InvertPermutationInvertPermutation=gradients_transpose_8_grad_invertpermutation_transpose_8_perm*
_output_shapes
:╣
$gradients/transpose_8_grad/transpose	Transpose)gradients/Reshape_7_grad/Reshape:output:00gradients/transpose_8_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
аау
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0*gradients/Reshape_13_grad/Reshape:output:0*gradients/Reshape_14_grad/Reshape:output:0*gradients/Reshape_15_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
N*
T0*
_output_shapes	
:ђ»
gradients/split_grad/concatConcatV2(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
N*
T0*
_output_shapes
:	ђ	Х
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0(gradients/transpose_7_grad/transpose:y:0(gradients/transpose_8_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
N*
T0* 
_output_shapes
:
ађ	\
gradients/concat_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :њ
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: f
gradients/concat_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:ђ	h
gradients/concat_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:ђ	╩
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0*
N* 
_output_shapes
::м
gradients/concat_grad/SliceSlice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:ђ	о
gradients/concat_grad/Slice_1Slice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:ђ	r
IdentityIdentity&gradients/transpose_grad/transpose:y:0*
T0*+
_output_shapes
:         u

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0*
T0*(
_output_shapes
:         аw

Identity_2Identity,gradients/ExpandDims_1_grad/Reshape:output:0*
T0*(
_output_shapes
:         аf

Identity_3Identity$gradients/split_grad/concat:output:0*
T0*
_output_shapes
:	ђ	i

Identity_4Identity&gradients/split_1_grad/concat:output:0*
T0* 
_output_shapes
:
ађ	d

Identity_5Identity&gradients/concat_grad/Slice_1:output:0*
T0*
_output_shapes	
:ђ	"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*(
_construction_contextkEagerRuntime*ѓ
_input_shapes­
ь:         а:         а:         а:         а: :         а:         а:         а:         а::         :         а:         а:ђ╗::         а:         а: ::::::::: : : : *=
api_implements+)lstm_e15ea421-5ae9-42eb-bcec-970c1aedc419*
api_preferred_deviceGPU*C
forward_function_name*(__forward_gpu_lstm_with_fallback_1208145*
go_backwards( *

time_major( :. *
(
_output_shapes
:         а:2.
,
_output_shapes
:         а:.*
(
_output_shapes
:         а:.*
(
_output_shapes
:         а:

_output_shapes
: :.*
(
_output_shapes
:         а:2.
,
_output_shapes
:         а:2.
,
_output_shapes
:         а:2.
,
_output_shapes
:         а:	

_output_shapes
::1
-
+
_output_shapes
:         :2.
,
_output_shapes
:         а:2.
,
_output_shapes
:         а:"

_output_shapes

:ђ╗: 

_output_shapes
::.*
(
_output_shapes
:         а:.*
(
_output_shapes
:         а:

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
┘

c
G__inference_reshape_13_layer_call_and_return_conditional_losses_1209524

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Л
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :Ј
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:h
ReshapeReshapeinputsReshape/shape:output:0*
T0*+
_output_shapes
:         \
IdentityIdentityReshape:output:0*
T0*+
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
┴B
═
*__inference_gpu_lstm_with_fallback_1209308

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4ѕc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:         P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : p

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0*,
_output_shapes
:         аR
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : t
ExpandDims_1
ExpandDimsinit_cExpandDims_1/dim:output:0*
T0*,
_output_shapes
:         аQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ї
splitSplitsplit/split_dim:output:0kernel*
T0*@
_output_shapes.
,:	а:	а:	а:	а*
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ъ
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*D
_output_shapes2
0:
аа:
аа:
аа:
аа*
	num_splite
zeros_like/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB:ђ	U
zeros_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    x

zeros_likeFill#zeros_like/shape_as_tensor:output:0zeros_like/Const:output:0*
T0*
_output_shapes	
:ђ	M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : r
concatConcatV2zeros_like:output:0biasconcat/axis:output:0*
N*
T0*
_output_shapes	
:ђS
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : Ц
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*L
_output_shapes:
8:а:а:а:а:а:а:а:а*
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
         a
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       m
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes
:	аY
ReshapeReshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:аa
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       m
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes
:	а[
	Reshape_1Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:аa
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       m
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes
:	а[
	Reshape_2Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:аa
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       m
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0*
_output_shapes
:	а[
	Reshape_3Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes	
:аa
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       p
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0* 
_output_shapes
:
аа\
	Reshape_4Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes

:ђѕa
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       p
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0* 
_output_shapes
:
аа\
	Reshape_5Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes

:ђѕa
transpose_7/permConst*
_output_shapes
:*
dtype0*
valueB"       p
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0* 
_output_shapes
:
аа\
	Reshape_6Reshapetranspose_7:y:0Const:output:0*
T0*
_output_shapes

:ђѕa
transpose_8/permConst*
_output_shapes
:*
dtype0*
valueB"       p
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0* 
_output_shapes
:
аа\
	Reshape_7Reshapetranspose_8:y:0Const:output:0*
T0*
_output_shapes

:ђѕ\
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes	
:а\
	Reshape_9Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes	
:а]

Reshape_10Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes	
:а]

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes	
:а]

Reshape_12Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes	
:а]

Reshape_13Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes	
:а]

Reshape_14Reshapesplit_2:output:6Const:output:0*
T0*
_output_shapes	
:а]

Reshape_15Reshapesplit_2:output:7Const:output:0*
T0*
_output_shapes	
:аO
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : а
concat_1ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T0*
_output_shapes

:ђ╗═
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1:output:0*
T0*`
_output_shapesN
L:         а:         а:         а:f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
         _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Т
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         а*
shrink_axis_maskq
SqueezeSqueezeCudnnRNN:output_h:0*
T0*(
_output_shapes
:         а*
squeeze_dims
 s
	Squeeze_1SqueezeCudnnRNN:output_c:0*
T0*(
_output_shapes
:         а*
squeeze_dims
 R
ExpandDims_2/dimConst*
_output_shapes
: *
dtype0*
value	B :ё
ExpandDims_2
ExpandDimsstrided_slice:output:0ExpandDims_2/dim:output:0*
T0*,
_output_shapes
:         а[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @_
IdentityIdentitystrided_slice:output:0*
T0*(
_output_shapes
:         аd

Identity_1IdentityExpandDims_2:output:0*
T0*,
_output_shapes
:         а[

Identity_2IdentitySqueeze:output:0*
T0*(
_output_shapes
:         а]

Identity_3IdentitySqueeze_1:output:0*
T0*(
_output_shapes
:         аI

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*p
_input_shapes_
]:         :         а:         а:	ђ	:
ађ	:ђ	*=
api_implements+)lstm_4726b1d6-6e07-45d8-b9f1-158de8b4230f*
api_preferred_deviceGPU*
go_backwards( *

time_major( :S O
+
_output_shapes
:         
 
_user_specified_nameinputs:PL
(
_output_shapes
:         а
 
_user_specified_nameinit_h:PL
(
_output_shapes
:         а
 
_user_specified_nameinit_c:GC

_output_shapes
:	ђ	
 
_user_specified_namekernel:RN
 
_output_shapes
:
ађ	
*
_user_specified_namerecurrent_kernel:A=

_output_shapes	
:ђ	

_user_specified_namebias
эM
б
(__forward_gpu_lstm_with_fallback_1209997

inputs
init_h_0
init_c_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4
strided_slice
cudnnrnn

cudnnrnn_0

cudnnrnn_1

cudnnrnn_2
	transpose

expanddims
expanddims_1
concat_1
transpose_perm

init_h

init_c
concat_1_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
transpose_7_perm
transpose_8_perm
split_2_split_dim
split_split_dim
split_1_split_dim
concat_axisѕc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          B
transpose_0	Transposeinputstranspose/perm:output:0*
T0P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : r

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T0*,
_output_shapes
:         аR
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : v
ExpandDims_1
ExpandDimsinit_c_0ExpandDims_1/dim:output:0*
T0*,
_output_shapes
:         аQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ї
splitSplitsplit/split_dim:output:0kernel*
T0*@
_output_shapes.
,:	а:	а:	а:	а*
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ъ
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*D
_output_shapes2
0:
аа:
аа:
аа:
аа*
	num_splite
zeros_like/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB:ђ	U
zeros_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    x

zeros_likeFill#zeros_like/shape_as_tensor:output:0zeros_like/Const:output:0*
T0*
_output_shapes	
:ђ	M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : r
concatConcatV2zeros_like:output:0biasconcat/axis:output:0*
N*
T0*
_output_shapes	
:ђS
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : Ц
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*L
_output_shapes:
8:а:а:а:а:а:а:а:а*
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
         a
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       m
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes
:	аY
ReshapeReshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:аa
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       m
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes
:	а[
	Reshape_1Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:аa
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       m
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes
:	а[
	Reshape_2Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:аa
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       m
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0*
_output_shapes
:	а[
	Reshape_3Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes	
:аa
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       p
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0* 
_output_shapes
:
аа\
	Reshape_4Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes

:ђѕa
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       p
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0* 
_output_shapes
:
аа\
	Reshape_5Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes

:ђѕa
transpose_7/permConst*
_output_shapes
:*
dtype0*
valueB"       p
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0* 
_output_shapes
:
аа\
	Reshape_6Reshapetranspose_7:y:0Const:output:0*
T0*
_output_shapes

:ђѕa
transpose_8/permConst*
_output_shapes
:*
dtype0*
valueB"       p
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0* 
_output_shapes
:
аа\
	Reshape_7Reshapetranspose_8:y:0Const:output:0*
T0*
_output_shapes

:ђѕ\
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes	
:а\
	Reshape_9Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes	
:а]

Reshape_10Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes	
:а]

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes	
:а]

Reshape_12Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes	
:а]

Reshape_13Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes	
:а]

Reshape_14Reshapesplit_2:output:6Const:output:0*
T0*
_output_shapes	
:а]

Reshape_15Reshapesplit_2:output:7Const:output:0*
T0*
_output_shapes	
:аO
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ё

concat_1_0ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T0Л
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1_0:output:0*
T0*`
_output_shapesN
L:         а:         а:         а:f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
         _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Й
strided_slice_0StridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
shrink_axis_maskq
SqueezeSqueezeCudnnRNN:output_h:0*
T0*(
_output_shapes
:         а*
squeeze_dims
 s
	Squeeze_1SqueezeCudnnRNN:output_c:0*
T0*(
_output_shapes
:         а*
squeeze_dims
 R
ExpandDims_2/dimConst*
_output_shapes
: *
dtype0*
value	B :є
ExpandDims_2
ExpandDimsstrided_slice_0:output:0ExpandDims_2/dim:output:0*
T0*,
_output_shapes
:         а[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @a
IdentityIdentitystrided_slice_0:output:0*
T0*(
_output_shapes
:         аd

Identity_1IdentityExpandDims_2:output:0*
T0*,
_output_shapes
:         а[

Identity_2IdentitySqueeze:output:0*
T0*(
_output_shapes
:         а]

Identity_3IdentitySqueeze_1:output:0*
T0*(
_output_shapes
:         аI

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: "
concat_1concat_1_0:output:0"'
concat_1_axisconcat_1/axis:output:0"#
concat_axisconcat/axis:output:0"
cudnnrnnCudnnRNN:output_h:0"!

cudnnrnn_0CudnnRNN:output_c:0"

cudnnrnn_1CudnnRNN:output:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"!

expanddimsExpandDims:output:0"%
expanddims_1ExpandDims_1:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
init_cinit_c_0"
init_hinit_h_0"/
split_1_split_dimsplit_1/split_dim:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"+
split_split_dimsplit/split_dim:output:0")
strided_slicestrided_slice_0:output:0"
	transposetranspose_0:y:0"-
transpose_1_permtranspose_1/perm:output:0"-
transpose_2_permtranspose_2/perm:output:0"-
transpose_3_permtranspose_3/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"-
transpose_5_permtranspose_5/perm:output:0"-
transpose_6_permtranspose_6/perm:output:0"-
transpose_7_permtranspose_7/perm:output:0"-
transpose_8_permtranspose_8/perm:output:0")
transpose_permtranspose/perm:output:0*(
_construction_contextkEagerRuntime*p
_input_shapes_
]:         :         а:         а:	ђ	:
ађ	:ђ	*=
api_implements+)lstm_6b3f42d6-117f-4a93-968a-eb0630b50933*
api_preferred_deviceGPU*Y
backward_function_name?=__inference___backward_gpu_lstm_with_fallback_1209822_1209998*
go_backwards( *

time_major( :S O
+
_output_shapes
:         
 
_user_specified_nameinputs:PL
(
_output_shapes
:         а
 
_user_specified_nameinit_h:PL
(
_output_shapes
:         а
 
_user_specified_nameinit_c:GC

_output_shapes
:	ђ	
 
_user_specified_namekernel:RN
 
_output_shapes
:
ађ	
*
_user_specified_namerecurrent_kernel:A=

_output_shapes	
:ђ	

_user_specified_namebias
џ	
к
while_cond_1213071
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice5
1while_while_cond_1213071___redundant_placeholder05
1while_while_cond_1213071___redundant_placeholder15
1while_while_cond_1213071___redundant_placeholder25
1while_while_cond_1213071___redundant_placeholder3
while_identity
`

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :         а:         а: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:         а:.*
(
_output_shapes
:         а:

_output_shapes
: :

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
џ	
к
while_cond_1211719
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice5
1while_while_cond_1211719___redundant_placeholder05
1while_while_cond_1211719___redundant_placeholder15
1while_while_cond_1211719___redundant_placeholder25
1while_while_cond_1211719___redundant_placeholder3
while_identity
`

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :         а:         а: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:         а:.*
(
_output_shapes
:         а:

_output_shapes
: :

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
№)
л
while_body_1212641
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_matmul_kernel_0%
!while_matmul_1_recurrent_kernel_0
while_biasadd_bias_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_matmul_kernel#
while_matmul_1_recurrent_kernel
while_biasadd_biasѕ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       д
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0њ
while/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while_matmul_kernel_0*
T0*(
_output_shapes
:         ђ	Ѓ
while/MatMul_1MatMulwhile_placeholder_2!while_matmul_1_recurrent_kernel_0*
T0*(
_output_shapes
:         ђ	w
	while/addAddV2while/MatMul:product:0while/MatMul_1:product:0*
T0*(
_output_shapes
:         ђ	p
while/BiasAddBiasAddwhile/add:z:0while_biasadd_bias_0*
T0*(
_output_shapes
:         ђ	W
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :╠
while/splitSplitwhile/split/split_dim:output:0while/BiasAdd:output:0*
T0*d
_output_shapesR
P:         а:         а:         а:         а*
	num_splita
while/SigmoidSigmoidwhile/split:output:0*
T0*(
_output_shapes
:         аc
while/Sigmoid_1Sigmoidwhile/split:output:1*
T0*(
_output_shapes
:         аm
	while/mulMulwhile/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:         а[

while/TanhTanhwhile/split:output:2*
T0*(
_output_shapes
:         аh
while/mul_1Mulwhile/Sigmoid:y:0while/Tanh:y:0*
T0*(
_output_shapes
:         аg
while/add_1AddV2while/mul:z:0while/mul_1:z:0*
T0*(
_output_shapes
:         аc
while/Sigmoid_2Sigmoidwhile/split:output:3*
T0*(
_output_shapes
:         аX
while/Tanh_1Tanhwhile/add_1:z:0*
T0*(
_output_shapes
:         аl
while/mul_2Mulwhile/Sigmoid_2:y:0while/Tanh_1:y:0*
T0*(
_output_shapes
:         аr
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : Я
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/mul_2:z:0*
_output_shapes
: *
element_dtype0:жУмO
while/add_2/yConst*
_output_shapes
: *
dtype0*
value	B :`
while/add_2AddV2while_placeholderwhile/add_2/y:output:0*
T0*
_output_shapes
: O
while/add_3/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_3AddV2while_while_loop_counterwhile/add_3/y:output:0*
T0*
_output_shapes
: L
while/IdentityIdentitywhile/add_3:z:0*
T0*
_output_shapes
: ]
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: N
while/Identity_2Identitywhile/add_2:z:0*
T0*
_output_shapes
: y
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: `
while/Identity_4Identitywhile/mul_2:z:0*
T0*(
_output_shapes
:         а`
while/Identity_5Identitywhile/add_1:z:0*
T0*(
_output_shapes
:         а"*
while_biasadd_biaswhile_biasadd_bias_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"D
while_matmul_1_recurrent_kernel!while_matmul_1_recurrent_kernel_0",
while_matmul_kernelwhile_matmul_kernel_0",
while_strided_slicewhile_strided_slice_0"е
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*e
_input_shapesT
R: : : : :         а:         а: : :	ђ	:
ађ	:ђ	: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:         а:.*
(
_output_shapes
:         а:

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	ђ	:&	"
 
_output_shapes
:
ађ	:!


_output_shapes	
:ђ	
№)
л
while_body_1207787
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_matmul_kernel_0%
!while_matmul_1_recurrent_kernel_0
while_biasadd_bias_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_matmul_kernel#
while_matmul_1_recurrent_kernel
while_biasadd_biasѕ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       д
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0њ
while/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while_matmul_kernel_0*
T0*(
_output_shapes
:         ђ	Ѓ
while/MatMul_1MatMulwhile_placeholder_2!while_matmul_1_recurrent_kernel_0*
T0*(
_output_shapes
:         ђ	w
	while/addAddV2while/MatMul:product:0while/MatMul_1:product:0*
T0*(
_output_shapes
:         ђ	p
while/BiasAddBiasAddwhile/add:z:0while_biasadd_bias_0*
T0*(
_output_shapes
:         ђ	W
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :╠
while/splitSplitwhile/split/split_dim:output:0while/BiasAdd:output:0*
T0*d
_output_shapesR
P:         а:         а:         а:         а*
	num_splita
while/SigmoidSigmoidwhile/split:output:0*
T0*(
_output_shapes
:         аc
while/Sigmoid_1Sigmoidwhile/split:output:1*
T0*(
_output_shapes
:         аm
	while/mulMulwhile/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:         а[

while/TanhTanhwhile/split:output:2*
T0*(
_output_shapes
:         аh
while/mul_1Mulwhile/Sigmoid:y:0while/Tanh:y:0*
T0*(
_output_shapes
:         аg
while/add_1AddV2while/mul:z:0while/mul_1:z:0*
T0*(
_output_shapes
:         аc
while/Sigmoid_2Sigmoidwhile/split:output:3*
T0*(
_output_shapes
:         аX
while/Tanh_1Tanhwhile/add_1:z:0*
T0*(
_output_shapes
:         аl
while/mul_2Mulwhile/Sigmoid_2:y:0while/Tanh_1:y:0*
T0*(
_output_shapes
:         аr
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : Я
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/mul_2:z:0*
_output_shapes
: *
element_dtype0:жУмO
while/add_2/yConst*
_output_shapes
: *
dtype0*
value	B :`
while/add_2AddV2while_placeholderwhile/add_2/y:output:0*
T0*
_output_shapes
: O
while/add_3/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_3AddV2while_while_loop_counterwhile/add_3/y:output:0*
T0*
_output_shapes
: L
while/IdentityIdentitywhile/add_3:z:0*
T0*
_output_shapes
: ]
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: N
while/Identity_2Identitywhile/add_2:z:0*
T0*
_output_shapes
: y
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: `
while/Identity_4Identitywhile/mul_2:z:0*
T0*(
_output_shapes
:         а`
while/Identity_5Identitywhile/add_1:z:0*
T0*(
_output_shapes
:         а"*
while_biasadd_biaswhile_biasadd_bias_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"D
while_matmul_1_recurrent_kernel!while_matmul_1_recurrent_kernel_0",
while_matmul_kernelwhile_matmul_kernel_0",
while_strided_slicewhile_strided_slice_0"е
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*e
_input_shapesT
R: : : : :         а:         а: : :	ђ	:
ађ	:ђ	: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:         а:.*
(
_output_shapes
:         а:

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	ђ	:&	"
 
_output_shapes
:
ађ	:!


_output_shapes	
:ђ	
№)
л
while_body_1209126
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_matmul_kernel_0%
!while_matmul_1_recurrent_kernel_0
while_biasadd_bias_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_matmul_kernel#
while_matmul_1_recurrent_kernel
while_biasadd_biasѕ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       д
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0њ
while/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while_matmul_kernel_0*
T0*(
_output_shapes
:         ђ	Ѓ
while/MatMul_1MatMulwhile_placeholder_2!while_matmul_1_recurrent_kernel_0*
T0*(
_output_shapes
:         ђ	w
	while/addAddV2while/MatMul:product:0while/MatMul_1:product:0*
T0*(
_output_shapes
:         ђ	p
while/BiasAddBiasAddwhile/add:z:0while_biasadd_bias_0*
T0*(
_output_shapes
:         ђ	W
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :╠
while/splitSplitwhile/split/split_dim:output:0while/BiasAdd:output:0*
T0*d
_output_shapesR
P:         а:         а:         а:         а*
	num_splita
while/SigmoidSigmoidwhile/split:output:0*
T0*(
_output_shapes
:         аc
while/Sigmoid_1Sigmoidwhile/split:output:1*
T0*(
_output_shapes
:         аm
	while/mulMulwhile/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:         а[

while/TanhTanhwhile/split:output:2*
T0*(
_output_shapes
:         аh
while/mul_1Mulwhile/Sigmoid:y:0while/Tanh:y:0*
T0*(
_output_shapes
:         аg
while/add_1AddV2while/mul:z:0while/mul_1:z:0*
T0*(
_output_shapes
:         аc
while/Sigmoid_2Sigmoidwhile/split:output:3*
T0*(
_output_shapes
:         аX
while/Tanh_1Tanhwhile/add_1:z:0*
T0*(
_output_shapes
:         аl
while/mul_2Mulwhile/Sigmoid_2:y:0while/Tanh_1:y:0*
T0*(
_output_shapes
:         аr
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : Я
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/mul_2:z:0*
_output_shapes
: *
element_dtype0:жУмO
while/add_2/yConst*
_output_shapes
: *
dtype0*
value	B :`
while/add_2AddV2while_placeholderwhile/add_2/y:output:0*
T0*
_output_shapes
: O
while/add_3/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_3AddV2while_while_loop_counterwhile/add_3/y:output:0*
T0*
_output_shapes
: L
while/IdentityIdentitywhile/add_3:z:0*
T0*
_output_shapes
: ]
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: N
while/Identity_2Identitywhile/add_2:z:0*
T0*
_output_shapes
: y
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: `
while/Identity_4Identitywhile/mul_2:z:0*
T0*(
_output_shapes
:         а`
while/Identity_5Identitywhile/add_1:z:0*
T0*(
_output_shapes
:         а"*
while_biasadd_biaswhile_biasadd_bias_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"D
while_matmul_1_recurrent_kernel!while_matmul_1_recurrent_kernel_0",
while_matmul_kernelwhile_matmul_kernel_0",
while_strided_slicewhile_strided_slice_0"е
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*e
_input_shapesT
R: : : : :         а:         а: : :	ђ	:
ађ	:ђ	: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:         а:.*
(
_output_shapes
:         а:

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	ђ	:&	"
 
_output_shapes
:
ађ	:!


_output_shapes	
:ђ	
№)
л
while_body_1213503
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_matmul_kernel_0%
!while_matmul_1_recurrent_kernel_0
while_biasadd_bias_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_matmul_kernel#
while_matmul_1_recurrent_kernel
while_biasadd_biasѕ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       д
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0њ
while/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while_matmul_kernel_0*
T0*(
_output_shapes
:         ђ	Ѓ
while/MatMul_1MatMulwhile_placeholder_2!while_matmul_1_recurrent_kernel_0*
T0*(
_output_shapes
:         ђ	w
	while/addAddV2while/MatMul:product:0while/MatMul_1:product:0*
T0*(
_output_shapes
:         ђ	p
while/BiasAddBiasAddwhile/add:z:0while_biasadd_bias_0*
T0*(
_output_shapes
:         ђ	W
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :╠
while/splitSplitwhile/split/split_dim:output:0while/BiasAdd:output:0*
T0*d
_output_shapesR
P:         а:         а:         а:         а*
	num_splita
while/SigmoidSigmoidwhile/split:output:0*
T0*(
_output_shapes
:         аc
while/Sigmoid_1Sigmoidwhile/split:output:1*
T0*(
_output_shapes
:         аm
	while/mulMulwhile/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:         а[

while/TanhTanhwhile/split:output:2*
T0*(
_output_shapes
:         аh
while/mul_1Mulwhile/Sigmoid:y:0while/Tanh:y:0*
T0*(
_output_shapes
:         аg
while/add_1AddV2while/mul:z:0while/mul_1:z:0*
T0*(
_output_shapes
:         аc
while/Sigmoid_2Sigmoidwhile/split:output:3*
T0*(
_output_shapes
:         аX
while/Tanh_1Tanhwhile/add_1:z:0*
T0*(
_output_shapes
:         аl
while/mul_2Mulwhile/Sigmoid_2:y:0while/Tanh_1:y:0*
T0*(
_output_shapes
:         аr
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : Я
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/mul_2:z:0*
_output_shapes
: *
element_dtype0:жУмO
while/add_2/yConst*
_output_shapes
: *
dtype0*
value	B :`
while/add_2AddV2while_placeholderwhile/add_2/y:output:0*
T0*
_output_shapes
: O
while/add_3/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_3AddV2while_while_loop_counterwhile/add_3/y:output:0*
T0*
_output_shapes
: L
while/IdentityIdentitywhile/add_3:z:0*
T0*
_output_shapes
: ]
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: N
while/Identity_2Identitywhile/add_2:z:0*
T0*
_output_shapes
: y
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: `
while/Identity_4Identitywhile/mul_2:z:0*
T0*(
_output_shapes
:         а`
while/Identity_5Identitywhile/add_1:z:0*
T0*(
_output_shapes
:         а"*
while_biasadd_biaswhile_biasadd_bias_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"D
while_matmul_1_recurrent_kernel!while_matmul_1_recurrent_kernel_0",
while_matmul_kernelwhile_matmul_kernel_0",
while_strided_slicewhile_strided_slice_0"е
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*e
_input_shapesT
R: : : : :         а:         а: : :	ђ	:
ађ	:ђ	: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:         а:.*
(
_output_shapes
:         а:

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	ђ	:&	"
 
_output_shapes
:
ађ	:!


_output_shapes	
:ђ	
к;
┴
!__inference_standard_lstm_1210883

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:         B
ShapeShapetranspose:y:0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Л
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ▓
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмє
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       Я
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУм_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ж
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maske
MatMulMatMulstrided_slice_1:output:0kernel*
T0*(
_output_shapes
:         ђ	_
MatMul_1MatMulinit_hrecurrent_kernel*
T0*(
_output_shapes
:         ђ	e
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         ђ	T
BiasAddBiasAddadd:z:0bias*
T0*(
_output_shapes
:         ђ	Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :║
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:         а:         а:         а:         а*
	num_splitU
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:         аW
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:         аT
mulMulSigmoid_1:y:0init_c*
T0*(
_output_shapes
:         аO
TanhTanhsplit:output:2*
T0*(
_output_shapes
:         аV
mul_1MulSigmoid:y:0Tanh:y:0*
T0*(
_output_shapes
:         аU
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:         аW
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:         аL
Tanh_1Tanh	add_1:z:0*
T0*(
_output_shapes
:         аZ
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*(
_output_shapes
:         аn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :┼
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Й
whileStatelessWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hinit_cstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelrecurrent_kernelbias*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*f
_output_shapesT
R: : : : :         а:         а: : :	ђ	:
ађ	:ђ	* 
_read_only_resource_inputs
 *
_stateful_parallelism( *
bodyR
while_body_1210797*
condR
while_cond_1210796*e
output_shapesT
R: : : : :         а:         а: : :	ђ	:
ађ	:ђ	*
parallel_iterations Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       О
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:         а*
element_dtype0*
num_elementsh
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ѕ
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         а*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ќ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:         а[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  ђ?a
IdentityIdentitystrided_slice_2:output:0*
T0*(
_output_shapes
:         а^

Identity_1Identitytranspose_1:y:0*
T0*,
_output_shapes
:         аY

Identity_2Identitywhile:output:4*
T0*(
_output_shapes
:         аY

Identity_3Identitywhile:output:5*
T0*(
_output_shapes
:         аI

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*p
_input_shapes_
]:         :         а:         а:	ђ	:
ађ	:ђ	*=
api_implements+)lstm_55c05eb7-11a9-4616-8925-c9051a66537d*
api_preferred_deviceCPU*
go_backwards( *

time_major( :S O
+
_output_shapes
:         
 
_user_specified_nameinputs:PL
(
_output_shapes
:         а
 
_user_specified_nameinit_h:PL
(
_output_shapes
:         а
 
_user_specified_nameinit_c:GC

_output_shapes
:	ђ	
 
_user_specified_namekernel:RN
 
_output_shapes
:
ађ	
*
_user_specified_namerecurrent_kernel:A=

_output_shapes	
:ђ	

_user_specified_namebias
№)
л
while_body_1210350
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_matmul_kernel_0%
!while_matmul_1_recurrent_kernel_0
while_biasadd_bias_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_matmul_kernel#
while_matmul_1_recurrent_kernel
while_biasadd_biasѕ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       д
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0њ
while/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while_matmul_kernel_0*
T0*(
_output_shapes
:         ђ	Ѓ
while/MatMul_1MatMulwhile_placeholder_2!while_matmul_1_recurrent_kernel_0*
T0*(
_output_shapes
:         ђ	w
	while/addAddV2while/MatMul:product:0while/MatMul_1:product:0*
T0*(
_output_shapes
:         ђ	p
while/BiasAddBiasAddwhile/add:z:0while_biasadd_bias_0*
T0*(
_output_shapes
:         ђ	W
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :╠
while/splitSplitwhile/split/split_dim:output:0while/BiasAdd:output:0*
T0*d
_output_shapesR
P:         а:         а:         а:         а*
	num_splita
while/SigmoidSigmoidwhile/split:output:0*
T0*(
_output_shapes
:         аc
while/Sigmoid_1Sigmoidwhile/split:output:1*
T0*(
_output_shapes
:         аm
	while/mulMulwhile/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:         а[

while/TanhTanhwhile/split:output:2*
T0*(
_output_shapes
:         аh
while/mul_1Mulwhile/Sigmoid:y:0while/Tanh:y:0*
T0*(
_output_shapes
:         аg
while/add_1AddV2while/mul:z:0while/mul_1:z:0*
T0*(
_output_shapes
:         аc
while/Sigmoid_2Sigmoidwhile/split:output:3*
T0*(
_output_shapes
:         аX
while/Tanh_1Tanhwhile/add_1:z:0*
T0*(
_output_shapes
:         аl
while/mul_2Mulwhile/Sigmoid_2:y:0while/Tanh_1:y:0*
T0*(
_output_shapes
:         аr
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : Я
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/mul_2:z:0*
_output_shapes
: *
element_dtype0:жУмO
while/add_2/yConst*
_output_shapes
: *
dtype0*
value	B :`
while/add_2AddV2while_placeholderwhile/add_2/y:output:0*
T0*
_output_shapes
: O
while/add_3/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_3AddV2while_while_loop_counterwhile/add_3/y:output:0*
T0*
_output_shapes
: L
while/IdentityIdentitywhile/add_3:z:0*
T0*
_output_shapes
: ]
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: N
while/Identity_2Identitywhile/add_2:z:0*
T0*
_output_shapes
: y
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: `
while/Identity_4Identitywhile/mul_2:z:0*
T0*(
_output_shapes
:         а`
while/Identity_5Identitywhile/add_1:z:0*
T0*(
_output_shapes
:         а"*
while_biasadd_biaswhile_biasadd_bias_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"D
while_matmul_1_recurrent_kernel!while_matmul_1_recurrent_kernel_0",
while_matmul_kernelwhile_matmul_kernel_0",
while_strided_slicewhile_strided_slice_0"е
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*e
_input_shapesT
R: : : : :         а:         а: : :	ђ	:
ађ	:ђ	: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:         а:.*
(
_output_shapes
:         а:

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	ђ	:&	"
 
_output_shapes
:
ађ	:!


_output_shapes	
:ђ	
тB
═
*__inference_gpu_lstm_with_fallback_1208420

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4ѕc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :                  P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : p

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0*,
_output_shapes
:         аR
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : t
ExpandDims_1
ExpandDimsinit_cExpandDims_1/dim:output:0*
T0*,
_output_shapes
:         аQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ї
splitSplitsplit/split_dim:output:0kernel*
T0*@
_output_shapes.
,:	а:	а:	а:	а*
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ъ
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*D
_output_shapes2
0:
аа:
аа:
аа:
аа*
	num_splite
zeros_like/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB:ђ	U
zeros_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    x

zeros_likeFill#zeros_like/shape_as_tensor:output:0zeros_like/Const:output:0*
T0*
_output_shapes	
:ђ	M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : r
concatConcatV2zeros_like:output:0biasconcat/axis:output:0*
N*
T0*
_output_shapes	
:ђS
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : Ц
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*L
_output_shapes:
8:а:а:а:а:а:а:а:а*
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
         a
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       m
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes
:	аY
ReshapeReshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:аa
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       m
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes
:	а[
	Reshape_1Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:аa
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       m
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes
:	а[
	Reshape_2Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:аa
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       m
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0*
_output_shapes
:	а[
	Reshape_3Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes	
:аa
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       p
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0* 
_output_shapes
:
аа\
	Reshape_4Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes

:ђѕa
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       p
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0* 
_output_shapes
:
аа\
	Reshape_5Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes

:ђѕa
transpose_7/permConst*
_output_shapes
:*
dtype0*
valueB"       p
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0* 
_output_shapes
:
аа\
	Reshape_6Reshapetranspose_7:y:0Const:output:0*
T0*
_output_shapes

:ђѕa
transpose_8/permConst*
_output_shapes
:*
dtype0*
valueB"       p
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0* 
_output_shapes
:
аа\
	Reshape_7Reshapetranspose_8:y:0Const:output:0*
T0*
_output_shapes

:ђѕ\
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes	
:а\
	Reshape_9Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes	
:а]

Reshape_10Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes	
:а]

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes	
:а]

Reshape_12Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes	
:а]

Reshape_13Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes	
:а]

Reshape_14Reshapesplit_2:output:6Const:output:0*
T0*
_output_shapes	
:а]

Reshape_15Reshapesplit_2:output:7Const:output:0*
T0*
_output_shapes	
:аO
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : а
concat_1ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T0*
_output_shapes

:ђ╗о
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1:output:0*
T0*i
_output_shapesW
U:                  а:         а:         а:f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
         _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Т
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         а*
shrink_axis_maskq
SqueezeSqueezeCudnnRNN:output_h:0*
T0*(
_output_shapes
:         а*
squeeze_dims
 s
	Squeeze_1SqueezeCudnnRNN:output_c:0*
T0*(
_output_shapes
:         а*
squeeze_dims
 R
ExpandDims_2/dimConst*
_output_shapes
: *
dtype0*
value	B :ё
ExpandDims_2
ExpandDimsstrided_slice:output:0ExpandDims_2/dim:output:0*
T0*,
_output_shapes
:         а[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @_
IdentityIdentitystrided_slice:output:0*
T0*(
_output_shapes
:         аd

Identity_1IdentityExpandDims_2:output:0*
T0*,
_output_shapes
:         а[

Identity_2IdentitySqueeze:output:0*
T0*(
_output_shapes
:         а]

Identity_3IdentitySqueeze_1:output:0*
T0*(
_output_shapes
:         аI

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*y
_input_shapesh
f:                  :         а:         а:	ђ	:
ађ	:ђ	*=
api_implements+)lstm_d272952e-0e4d-4b32-9764-a4b66b442161*
api_preferred_deviceGPU*
go_backwards( *

time_major( :\ X
4
_output_shapes"
 :                  
 
_user_specified_nameinputs:PL
(
_output_shapes
:         а
 
_user_specified_nameinit_h:PL
(
_output_shapes
:         а
 
_user_specified_nameinit_c:GC

_output_shapes
:	ђ	
 
_user_specified_namekernel:RN
 
_output_shapes
:
ађ	
*
_user_specified_namerecurrent_kernel:A=

_output_shapes	
:ђ	

_user_specified_namebias
╠	
э
E__inference_dense_13_layer_call_and_return_conditional_losses_1209505

inputs1
matmul_readvariableop_resource:	а-
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	а*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         а: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         а
 
_user_specified_nameinputs
№)
л
while_body_1213072
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_matmul_kernel_0%
!while_matmul_1_recurrent_kernel_0
while_biasadd_bias_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_matmul_kernel#
while_matmul_1_recurrent_kernel
while_biasadd_biasѕ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       д
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0њ
while/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while_matmul_kernel_0*
T0*(
_output_shapes
:         ђ	Ѓ
while/MatMul_1MatMulwhile_placeholder_2!while_matmul_1_recurrent_kernel_0*
T0*(
_output_shapes
:         ђ	w
	while/addAddV2while/MatMul:product:0while/MatMul_1:product:0*
T0*(
_output_shapes
:         ђ	p
while/BiasAddBiasAddwhile/add:z:0while_biasadd_bias_0*
T0*(
_output_shapes
:         ђ	W
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :╠
while/splitSplitwhile/split/split_dim:output:0while/BiasAdd:output:0*
T0*d
_output_shapesR
P:         а:         а:         а:         а*
	num_splita
while/SigmoidSigmoidwhile/split:output:0*
T0*(
_output_shapes
:         аc
while/Sigmoid_1Sigmoidwhile/split:output:1*
T0*(
_output_shapes
:         аm
	while/mulMulwhile/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:         а[

while/TanhTanhwhile/split:output:2*
T0*(
_output_shapes
:         аh
while/mul_1Mulwhile/Sigmoid:y:0while/Tanh:y:0*
T0*(
_output_shapes
:         аg
while/add_1AddV2while/mul:z:0while/mul_1:z:0*
T0*(
_output_shapes
:         аc
while/Sigmoid_2Sigmoidwhile/split:output:3*
T0*(
_output_shapes
:         аX
while/Tanh_1Tanhwhile/add_1:z:0*
T0*(
_output_shapes
:         аl
while/mul_2Mulwhile/Sigmoid_2:y:0while/Tanh_1:y:0*
T0*(
_output_shapes
:         аr
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : Я
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/mul_2:z:0*
_output_shapes
: *
element_dtype0:жУмO
while/add_2/yConst*
_output_shapes
: *
dtype0*
value	B :`
while/add_2AddV2while_placeholderwhile/add_2/y:output:0*
T0*
_output_shapes
: O
while/add_3/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_3AddV2while_while_loop_counterwhile/add_3/y:output:0*
T0*
_output_shapes
: L
while/IdentityIdentitywhile/add_3:z:0*
T0*
_output_shapes
: ]
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: N
while/Identity_2Identitywhile/add_2:z:0*
T0*
_output_shapes
: y
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: `
while/Identity_4Identitywhile/mul_2:z:0*
T0*(
_output_shapes
:         а`
while/Identity_5Identitywhile/add_1:z:0*
T0*(
_output_shapes
:         а"*
while_biasadd_biaswhile_biasadd_bias_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"D
while_matmul_1_recurrent_kernel!while_matmul_1_recurrent_kernel_0",
while_matmul_kernelwhile_matmul_kernel_0",
while_strided_slicewhile_strided_slice_0"е
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*e
_input_shapesT
R: : : : :         а:         а: : :	ђ	:
ађ	:ђ	: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:         а:.*
(
_output_shapes
:         а:

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	ђ	:&	"
 
_output_shapes
:
ађ	:!


_output_shapes	
:ђ	
у
Ш
1__inference_change_layer_12_layer_call_fn_1210265

inputs
unknown:	ђ	
	unknown_0:
ађ	
	unknown_1:	ђ	
	unknown_2:	а
	unknown_3:
identityѕбStatefulPartitionedCallї
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         *'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8ѓ *U
fPRN
L__inference_change_layer_12_layer_call_and_return_conditional_losses_1210125s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:         : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
№)
л
while_body_1211274
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_matmul_kernel_0%
!while_matmul_1_recurrent_kernel_0
while_biasadd_bias_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_matmul_kernel#
while_matmul_1_recurrent_kernel
while_biasadd_biasѕ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       д
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0њ
while/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while_matmul_kernel_0*
T0*(
_output_shapes
:         ђ	Ѓ
while/MatMul_1MatMulwhile_placeholder_2!while_matmul_1_recurrent_kernel_0*
T0*(
_output_shapes
:         ђ	w
	while/addAddV2while/MatMul:product:0while/MatMul_1:product:0*
T0*(
_output_shapes
:         ђ	p
while/BiasAddBiasAddwhile/add:z:0while_biasadd_bias_0*
T0*(
_output_shapes
:         ђ	W
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :╠
while/splitSplitwhile/split/split_dim:output:0while/BiasAdd:output:0*
T0*d
_output_shapesR
P:         а:         а:         а:         а*
	num_splita
while/SigmoidSigmoidwhile/split:output:0*
T0*(
_output_shapes
:         аc
while/Sigmoid_1Sigmoidwhile/split:output:1*
T0*(
_output_shapes
:         аm
	while/mulMulwhile/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:         а[

while/TanhTanhwhile/split:output:2*
T0*(
_output_shapes
:         аh
while/mul_1Mulwhile/Sigmoid:y:0while/Tanh:y:0*
T0*(
_output_shapes
:         аg
while/add_1AddV2while/mul:z:0while/mul_1:z:0*
T0*(
_output_shapes
:         аc
while/Sigmoid_2Sigmoidwhile/split:output:3*
T0*(
_output_shapes
:         аX
while/Tanh_1Tanhwhile/add_1:z:0*
T0*(
_output_shapes
:         аl
while/mul_2Mulwhile/Sigmoid_2:y:0while/Tanh_1:y:0*
T0*(
_output_shapes
:         аr
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : Я
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/mul_2:z:0*
_output_shapes
: *
element_dtype0:жУмO
while/add_2/yConst*
_output_shapes
: *
dtype0*
value	B :`
while/add_2AddV2while_placeholderwhile/add_2/y:output:0*
T0*
_output_shapes
: O
while/add_3/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_3AddV2while_while_loop_counterwhile/add_3/y:output:0*
T0*
_output_shapes
: L
while/IdentityIdentitywhile/add_3:z:0*
T0*
_output_shapes
: ]
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: N
while/Identity_2Identitywhile/add_2:z:0*
T0*
_output_shapes
: y
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: `
while/Identity_4Identitywhile/mul_2:z:0*
T0*(
_output_shapes
:         а`
while/Identity_5Identitywhile/add_1:z:0*
T0*(
_output_shapes
:         а"*
while_biasadd_biaswhile_biasadd_bias_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"D
while_matmul_1_recurrent_kernel!while_matmul_1_recurrent_kernel_0",
while_matmul_kernelwhile_matmul_kernel_0",
while_strided_slicewhile_strided_slice_0"е
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*e
_input_shapesT
R: : : : :         а:         а: : :	ђ	:
ађ	:ђ	: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:         а:.*
(
_output_shapes
:         а:

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	ђ	:&	"
 
_output_shapes
:
ађ	:!


_output_shapes	
:ђ	
эM
б
(__forward_gpu_lstm_with_fallback_1213430

inputs
init_h_0
init_c_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4
strided_slice
cudnnrnn

cudnnrnn_0

cudnnrnn_1

cudnnrnn_2
	transpose

expanddims
expanddims_1
concat_1
transpose_perm

init_h

init_c
concat_1_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
transpose_7_perm
transpose_8_perm
split_2_split_dim
split_split_dim
split_1_split_dim
concat_axisѕc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          B
transpose_0	Transposeinputstranspose/perm:output:0*
T0P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : r

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T0*,
_output_shapes
:         аR
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : v
ExpandDims_1
ExpandDimsinit_c_0ExpandDims_1/dim:output:0*
T0*,
_output_shapes
:         аQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ї
splitSplitsplit/split_dim:output:0kernel*
T0*@
_output_shapes.
,:	а:	а:	а:	а*
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ъ
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*D
_output_shapes2
0:
аа:
аа:
аа:
аа*
	num_splite
zeros_like/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB:ђ	U
zeros_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    x

zeros_likeFill#zeros_like/shape_as_tensor:output:0zeros_like/Const:output:0*
T0*
_output_shapes	
:ђ	M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : r
concatConcatV2zeros_like:output:0biasconcat/axis:output:0*
N*
T0*
_output_shapes	
:ђS
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : Ц
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*L
_output_shapes:
8:а:а:а:а:а:а:а:а*
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
         a
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       m
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes
:	аY
ReshapeReshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:аa
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       m
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes
:	а[
	Reshape_1Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:аa
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       m
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes
:	а[
	Reshape_2Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:аa
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       m
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0*
_output_shapes
:	а[
	Reshape_3Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes	
:аa
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       p
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0* 
_output_shapes
:
аа\
	Reshape_4Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes

:ђѕa
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       p
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0* 
_output_shapes
:
аа\
	Reshape_5Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes

:ђѕa
transpose_7/permConst*
_output_shapes
:*
dtype0*
valueB"       p
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0* 
_output_shapes
:
аа\
	Reshape_6Reshapetranspose_7:y:0Const:output:0*
T0*
_output_shapes

:ђѕa
transpose_8/permConst*
_output_shapes
:*
dtype0*
valueB"       p
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0* 
_output_shapes
:
аа\
	Reshape_7Reshapetranspose_8:y:0Const:output:0*
T0*
_output_shapes

:ђѕ\
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes	
:а\
	Reshape_9Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes	
:а]

Reshape_10Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes	
:а]

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes	
:а]

Reshape_12Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes	
:а]

Reshape_13Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes	
:а]

Reshape_14Reshapesplit_2:output:6Const:output:0*
T0*
_output_shapes	
:а]

Reshape_15Reshapesplit_2:output:7Const:output:0*
T0*
_output_shapes	
:аO
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ё

concat_1_0ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T0Л
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1_0:output:0*
T0*`
_output_shapesN
L:         а:         а:         а:f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
         _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Й
strided_slice_0StridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
shrink_axis_maskq
SqueezeSqueezeCudnnRNN:output_h:0*
T0*(
_output_shapes
:         а*
squeeze_dims
 s
	Squeeze_1SqueezeCudnnRNN:output_c:0*
T0*(
_output_shapes
:         а*
squeeze_dims
 R
ExpandDims_2/dimConst*
_output_shapes
: *
dtype0*
value	B :є
ExpandDims_2
ExpandDimsstrided_slice_0:output:0ExpandDims_2/dim:output:0*
T0*,
_output_shapes
:         а[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @a
IdentityIdentitystrided_slice_0:output:0*
T0*(
_output_shapes
:         аd

Identity_1IdentityExpandDims_2:output:0*
T0*,
_output_shapes
:         а[

Identity_2IdentitySqueeze:output:0*
T0*(
_output_shapes
:         а]

Identity_3IdentitySqueeze_1:output:0*
T0*(
_output_shapes
:         аI

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: "
concat_1concat_1_0:output:0"'
concat_1_axisconcat_1/axis:output:0"#
concat_axisconcat/axis:output:0"
cudnnrnnCudnnRNN:output_h:0"!

cudnnrnn_0CudnnRNN:output_c:0"

cudnnrnn_1CudnnRNN:output:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"!

expanddimsExpandDims:output:0"%
expanddims_1ExpandDims_1:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
init_cinit_c_0"
init_hinit_h_0"/
split_1_split_dimsplit_1/split_dim:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"+
split_split_dimsplit/split_dim:output:0")
strided_slicestrided_slice_0:output:0"
	transposetranspose_0:y:0"-
transpose_1_permtranspose_1/perm:output:0"-
transpose_2_permtranspose_2/perm:output:0"-
transpose_3_permtranspose_3/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"-
transpose_5_permtranspose_5/perm:output:0"-
transpose_6_permtranspose_6/perm:output:0"-
transpose_7_permtranspose_7/perm:output:0"-
transpose_8_permtranspose_8/perm:output:0")
transpose_permtranspose/perm:output:0*(
_construction_contextkEagerRuntime*p
_input_shapes_
]:         :         а:         а:	ђ	:
ађ	:ђ	*=
api_implements+)lstm_27cab513-e281-4aef-8d50-ee58fd8ba0ea*
api_preferred_deviceGPU*Y
backward_function_name?=__inference___backward_gpu_lstm_with_fallback_1213255_1213431*
go_backwards( *

time_major( :S O
+
_output_shapes
:         
 
_user_specified_nameinputs:PL
(
_output_shapes
:         а
 
_user_specified_nameinit_h:PL
(
_output_shapes
:         а
 
_user_specified_nameinit_c:GC

_output_shapes
:	ђ	
 
_user_specified_namekernel:RN
 
_output_shapes
:
ађ	
*
_user_specified_namerecurrent_kernel:A=

_output_shapes	
:ђ	

_user_specified_namebias
ф┬
у
=__inference___backward_gpu_lstm_with_fallback_1209309_1209485
placeholder
placeholder_1
placeholder_2
placeholder_3
placeholder_43
/gradients_expanddims_2_grad_shape_strided_slice)
%gradients_squeeze_grad_shape_cudnnrnn+
'gradients_squeeze_1_grad_shape_cudnnrnn/
+gradients_strided_slice_grad_shape_cudnnrnn!
gradients_zeros_like_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9
5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_15
1gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h,
(gradients_expanddims_1_grad_shape_init_c-
)gradients_concat_1_grad_mod_concat_1_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_permA
=gradients_transpose_7_grad_invertpermutation_transpose_7_permA
=gradients_transpose_8_grad_invertpermutation_transpose_8_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim)
%gradients_concat_grad_mod_concat_axis
identity

identity_1

identity_2

identity_3

identity_4

identity_5ѕ_
gradients/grad_ys_0Identityplaceholder*
T0*(
_output_shapes
:         аe
gradients/grad_ys_1Identityplaceholder_1*
T0*,
_output_shapes
:         аa
gradients/grad_ys_2Identityplaceholder_2*
T0*(
_output_shapes
:         аa
gradients/grad_ys_3Identityplaceholder_3*
T0*(
_output_shapes
:         аO
gradients/grad_ys_4Identityplaceholder_4*
T0*
_output_shapes
: ђ
!gradients/ExpandDims_2_grad/ShapeShape/gradients_expanddims_2_grad_shape_strided_slice*
T0*
_output_shapes
:Ф
#gradients/ExpandDims_2_grad/ReshapeReshapegradients/grad_ys_1:output:0*gradients/ExpandDims_2_grad/Shape:output:0*
T0*(
_output_shapes
:         аq
gradients/Squeeze_grad/ShapeShape%gradients_squeeze_grad_shape_cudnnrnn*
T0*
_output_shapes
:Ц
gradients/Squeeze_grad/ReshapeReshapegradients/grad_ys_2:output:0%gradients/Squeeze_grad/Shape:output:0*
T0*,
_output_shapes
:         аu
gradients/Squeeze_1_grad/ShapeShape'gradients_squeeze_1_grad_shape_cudnnrnn*
T0*
_output_shapes
:Е
 gradients/Squeeze_1_grad/ReshapeReshapegradients/grad_ys_3:output:0'gradients/Squeeze_1_grad/Shape:output:0*
T0*,
_output_shapes
:         ак
gradients/AddNAddNgradients/grad_ys_0:output:0,gradients/ExpandDims_2_grad/Reshape:output:0*
N*
T0*&
_class
loc:@gradients/grad_ys_0*(
_output_shapes
:         а}
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
:є
3gradients/strided_slice_grad/StridedSliceGrad/beginConst*
_output_shapes
:*
dtype0*
valueB:
         {
1gradients/strided_slice_grad/StridedSliceGrad/endConst*
_output_shapes
:*
dtype0*
valueB: 
5gradients/strided_slice_grad/StridedSliceGrad/stridesConst*
_output_shapes
:*
dtype0*
valueB:џ
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0<gradients/strided_slice_grad/StridedSliceGrad/begin:output:0:gradients/strided_slice_grad/StridedSliceGrad/end:output:0>gradients/strided_slice_grad/StridedSliceGrad/strides:output:0gradients/AddN:sum:0*
Index0*
T0*,
_output_shapes
:         а*
shrink_axis_maskc
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
:Ц
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_11gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1+gradients_strided_slice_grad_shape_cudnnrnn%gradients_squeeze_grad_shape_cudnnrnn'gradients_squeeze_1_grad_shape_cudnnrnn6gradients/strided_slice_grad/StridedSliceGrad:output:0'gradients/Squeeze_grad/Reshape:output:0)gradients/Squeeze_1_grad/Reshape:output:0gradients_zeros_like_cudnnrnn*
T0*c
_output_shapesQ
O:         :         а:         а:ђ╗ќ
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:л
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*+
_output_shapes
:         u
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
:к
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*(
_output_shapes
:         аy
!gradients/ExpandDims_1_grad/ShapeShape(gradients_expanddims_1_grad_shape_init_c*
T0*
_output_shapes
:╩
#gradients/ExpandDims_1_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_c_backprop:0*gradients/ExpandDims_1_grad/Shape:output:0*
T0*(
_output_shapes
:         а^
gradients/concat_1_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :џ
gradients/concat_1_grad/modFloorMod)gradients_concat_1_grad_mod_concat_1_axis%gradients/concat_1_grad/Rank:output:0*
T0*
_output_shapes
: h
gradients/concat_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:аj
gradients/concat_1_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:аj
gradients/concat_1_grad/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:аj
gradients/concat_1_grad/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:аk
gradients/concat_1_grad/Shape_4Const*
_output_shapes
:*
dtype0*
valueB:ђѕk
gradients/concat_1_grad/Shape_5Const*
_output_shapes
:*
dtype0*
valueB:ђѕk
gradients/concat_1_grad/Shape_6Const*
_output_shapes
:*
dtype0*
valueB:ђѕk
gradients/concat_1_grad/Shape_7Const*
_output_shapes
:*
dtype0*
valueB:ђѕj
gradients/concat_1_grad/Shape_8Const*
_output_shapes
:*
dtype0*
valueB:аj
gradients/concat_1_grad/Shape_9Const*
_output_shapes
:*
dtype0*
valueB:аk
 gradients/concat_1_grad/Shape_10Const*
_output_shapes
:*
dtype0*
valueB:аk
 gradients/concat_1_grad/Shape_11Const*
_output_shapes
:*
dtype0*
valueB:аk
 gradients/concat_1_grad/Shape_12Const*
_output_shapes
:*
dtype0*
valueB:аk
 gradients/concat_1_grad/Shape_13Const*
_output_shapes
:*
dtype0*
valueB:аk
 gradients/concat_1_grad/Shape_14Const*
_output_shapes
:*
dtype0*
valueB:аk
 gradients/concat_1_grad/Shape_15Const*
_output_shapes
:*
dtype0*
valueB:аЭ
$gradients/concat_1_grad/ConcatOffsetConcatOffsetgradients/concat_1_grad/mod:z:0&gradients/concat_1_grad/Shape:output:0(gradients/concat_1_grad/Shape_1:output:0(gradients/concat_1_grad/Shape_2:output:0(gradients/concat_1_grad/Shape_3:output:0(gradients/concat_1_grad/Shape_4:output:0(gradients/concat_1_grad/Shape_5:output:0(gradients/concat_1_grad/Shape_6:output:0(gradients/concat_1_grad/Shape_7:output:0(gradients/concat_1_grad/Shape_8:output:0(gradients/concat_1_grad/Shape_9:output:0)gradients/concat_1_grad/Shape_10:output:0)gradients/concat_1_grad/Shape_11:output:0)gradients/concat_1_grad/Shape_12:output:0)gradients/concat_1_grad/Shape_13:output:0)gradients/concat_1_grad/Shape_14:output:0)gradients/concat_1_grad/Shape_15:output:0*
N*t
_output_shapesb
`::::::::::::::::В
gradients/concat_1_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:0&gradients/concat_1_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:а­
gradients/concat_1_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:1(gradients/concat_1_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:а­
gradients/concat_1_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:2(gradients/concat_1_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes	
:а­
gradients/concat_1_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:3(gradients/concat_1_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes	
:аы
gradients/concat_1_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:4(gradients/concat_1_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes

:ђѕы
gradients/concat_1_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:5(gradients/concat_1_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes

:ђѕы
gradients/concat_1_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:6(gradients/concat_1_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes

:ђѕы
gradients/concat_1_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:7(gradients/concat_1_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes

:ђѕ­
gradients/concat_1_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:8(gradients/concat_1_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes	
:а­
gradients/concat_1_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:9(gradients/concat_1_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes	
:аз
 gradients/concat_1_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:10)gradients/concat_1_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes	
:аз
 gradients/concat_1_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:11)gradients/concat_1_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes	
:аз
 gradients/concat_1_grad/Slice_12Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:12)gradients/concat_1_grad/Shape_12:output:0*
Index0*
T0*
_output_shapes	
:аз
 gradients/concat_1_grad/Slice_13Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:13)gradients/concat_1_grad/Shape_13:output:0*
Index0*
T0*
_output_shapes	
:аз
 gradients/concat_1_grad/Slice_14Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:14)gradients/concat_1_grad/Shape_14:output:0*
Index0*
T0*
_output_shapes	
:аз
 gradients/concat_1_grad/Slice_15Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:15)gradients/concat_1_grad/Shape_15:output:0*
Index0*
T0*
_output_shapes	
:аm
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      б
gradients/Reshape_grad/ReshapeReshape&gradients/concat_1_grad/Slice:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes
:	аo
gradients/Reshape_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      е
 gradients/Reshape_1_grad/ReshapeReshape(gradients/concat_1_grad/Slice_1:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0*
_output_shapes
:	аo
gradients/Reshape_2_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      е
 gradients/Reshape_2_grad/ReshapeReshape(gradients/concat_1_grad/Slice_2:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0*
_output_shapes
:	аo
gradients/Reshape_3_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      е
 gradients/Reshape_3_grad/ReshapeReshape(gradients/concat_1_grad/Slice_3:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0*
_output_shapes
:	аo
gradients/Reshape_4_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      Е
 gradients/Reshape_4_grad/ReshapeReshape(gradients/concat_1_grad/Slice_4:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0* 
_output_shapes
:
ааo
gradients/Reshape_5_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      Е
 gradients/Reshape_5_grad/ReshapeReshape(gradients/concat_1_grad/Slice_5:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0* 
_output_shapes
:
ааo
gradients/Reshape_6_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      Е
 gradients/Reshape_6_grad/ReshapeReshape(gradients/concat_1_grad/Slice_6:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0* 
_output_shapes
:
ааo
gradients/Reshape_7_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      Е
 gradients/Reshape_7_grad/ReshapeReshape(gradients/concat_1_grad/Slice_7:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0* 
_output_shapes
:
ааi
gradients/Reshape_8_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:ац
 gradients/Reshape_8_grad/ReshapeReshape(gradients/concat_1_grad/Slice_8:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes	
:аi
gradients/Reshape_9_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:ац
 gradients/Reshape_9_grad/ReshapeReshape(gradients/concat_1_grad/Slice_9:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes	
:аj
gradients/Reshape_10_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:аД
!gradients/Reshape_10_grad/ReshapeReshape)gradients/concat_1_grad/Slice_10:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes	
:аj
gradients/Reshape_11_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:аД
!gradients/Reshape_11_grad/ReshapeReshape)gradients/concat_1_grad/Slice_11:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes	
:аj
gradients/Reshape_12_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:аД
!gradients/Reshape_12_grad/ReshapeReshape)gradients/concat_1_grad/Slice_12:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes	
:аj
gradients/Reshape_13_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:аД
!gradients/Reshape_13_grad/ReshapeReshape)gradients/concat_1_grad/Slice_13:output:0(gradients/Reshape_13_grad/Shape:output:0*
T0*
_output_shapes	
:аj
gradients/Reshape_14_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:аД
!gradients/Reshape_14_grad/ReshapeReshape)gradients/concat_1_grad/Slice_14:output:0(gradients/Reshape_14_grad/Shape:output:0*
T0*
_output_shapes	
:аj
gradients/Reshape_15_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:аД
!gradients/Reshape_15_grad/ReshapeReshape)gradients/concat_1_grad/Slice_15:output:0(gradients/Reshape_15_grad/Shape:output:0*
T0*
_output_shapes	
:аю
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:Х
$gradients/transpose_1_grad/transpose	Transpose'gradients/Reshape_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0*
_output_shapes
:	аю
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:И
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0*
_output_shapes
:	аю
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:И
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0*
_output_shapes
:	аю
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:И
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0*
_output_shapes
:	аю
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:╣
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
ааю
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:╣
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
ааю
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:╣
$gradients/transpose_7_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
ааю
,gradients/transpose_8_grad/InvertPermutationInvertPermutation=gradients_transpose_8_grad_invertpermutation_transpose_8_perm*
_output_shapes
:╣
$gradients/transpose_8_grad/transpose	Transpose)gradients/Reshape_7_grad/Reshape:output:00gradients/transpose_8_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
аау
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0*gradients/Reshape_13_grad/Reshape:output:0*gradients/Reshape_14_grad/Reshape:output:0*gradients/Reshape_15_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
N*
T0*
_output_shapes	
:ђ»
gradients/split_grad/concatConcatV2(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
N*
T0*
_output_shapes
:	ђ	Х
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0(gradients/transpose_7_grad/transpose:y:0(gradients/transpose_8_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
N*
T0* 
_output_shapes
:
ађ	\
gradients/concat_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :њ
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: f
gradients/concat_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:ђ	h
gradients/concat_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:ђ	╩
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0*
N* 
_output_shapes
::м
gradients/concat_grad/SliceSlice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:ђ	о
gradients/concat_grad/Slice_1Slice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:ђ	r
IdentityIdentity&gradients/transpose_grad/transpose:y:0*
T0*+
_output_shapes
:         u

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0*
T0*(
_output_shapes
:         аw

Identity_2Identity,gradients/ExpandDims_1_grad/Reshape:output:0*
T0*(
_output_shapes
:         аf

Identity_3Identity$gradients/split_grad/concat:output:0*
T0*
_output_shapes
:	ђ	i

Identity_4Identity&gradients/split_1_grad/concat:output:0*
T0* 
_output_shapes
:
ађ	d

Identity_5Identity&gradients/concat_grad/Slice_1:output:0*
T0*
_output_shapes	
:ђ	"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*(
_construction_contextkEagerRuntime*ѓ
_input_shapes­
ь:         а:         а:         а:         а: :         а:         а:         а:         а::         :         а:         а:ђ╗::         а:         а: ::::::::: : : : *=
api_implements+)lstm_4726b1d6-6e07-45d8-b9f1-158de8b4230f*
api_preferred_deviceGPU*C
forward_function_name*(__forward_gpu_lstm_with_fallback_1209484*
go_backwards( *

time_major( :. *
(
_output_shapes
:         а:2.
,
_output_shapes
:         а:.*
(
_output_shapes
:         а:.*
(
_output_shapes
:         а:

_output_shapes
: :.*
(
_output_shapes
:         а:2.
,
_output_shapes
:         а:2.
,
_output_shapes
:         а:2.
,
_output_shapes
:         а:	

_output_shapes
::1
-
+
_output_shapes
:         :2.
,
_output_shapes
:         а:2.
,
_output_shapes
:         а:"

_output_shapes

:ђ╗: 

_output_shapes
::.*
(
_output_shapes
:         а:.*
(
_output_shapes
:         а:

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Ы┬
у
=__inference___backward_gpu_lstm_with_fallback_1212824_1213000
placeholder
placeholder_1
placeholder_2
placeholder_3
placeholder_43
/gradients_expanddims_2_grad_shape_strided_slice)
%gradients_squeeze_grad_shape_cudnnrnn+
'gradients_squeeze_1_grad_shape_cudnnrnn/
+gradients_strided_slice_grad_shape_cudnnrnn!
gradients_zeros_like_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9
5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_15
1gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h,
(gradients_expanddims_1_grad_shape_init_c-
)gradients_concat_1_grad_mod_concat_1_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_permA
=gradients_transpose_7_grad_invertpermutation_transpose_7_permA
=gradients_transpose_8_grad_invertpermutation_transpose_8_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim)
%gradients_concat_grad_mod_concat_axis
identity

identity_1

identity_2

identity_3

identity_4

identity_5ѕ_
gradients/grad_ys_0Identityplaceholder*
T0*(
_output_shapes
:         аe
gradients/grad_ys_1Identityplaceholder_1*
T0*,
_output_shapes
:         аa
gradients/grad_ys_2Identityplaceholder_2*
T0*(
_output_shapes
:         аa
gradients/grad_ys_3Identityplaceholder_3*
T0*(
_output_shapes
:         аO
gradients/grad_ys_4Identityplaceholder_4*
T0*
_output_shapes
: ђ
!gradients/ExpandDims_2_grad/ShapeShape/gradients_expanddims_2_grad_shape_strided_slice*
T0*
_output_shapes
:Ф
#gradients/ExpandDims_2_grad/ReshapeReshapegradients/grad_ys_1:output:0*gradients/ExpandDims_2_grad/Shape:output:0*
T0*(
_output_shapes
:         аq
gradients/Squeeze_grad/ShapeShape%gradients_squeeze_grad_shape_cudnnrnn*
T0*
_output_shapes
:Ц
gradients/Squeeze_grad/ReshapeReshapegradients/grad_ys_2:output:0%gradients/Squeeze_grad/Shape:output:0*
T0*,
_output_shapes
:         аu
gradients/Squeeze_1_grad/ShapeShape'gradients_squeeze_1_grad_shape_cudnnrnn*
T0*
_output_shapes
:Е
 gradients/Squeeze_1_grad/ReshapeReshapegradients/grad_ys_3:output:0'gradients/Squeeze_1_grad/Shape:output:0*
T0*,
_output_shapes
:         ак
gradients/AddNAddNgradients/grad_ys_0:output:0,gradients/ExpandDims_2_grad/Reshape:output:0*
N*
T0*&
_class
loc:@gradients/grad_ys_0*(
_output_shapes
:         а}
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
:є
3gradients/strided_slice_grad/StridedSliceGrad/beginConst*
_output_shapes
:*
dtype0*
valueB:
         {
1gradients/strided_slice_grad/StridedSliceGrad/endConst*
_output_shapes
:*
dtype0*
valueB: 
5gradients/strided_slice_grad/StridedSliceGrad/stridesConst*
_output_shapes
:*
dtype0*
valueB:Б
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0<gradients/strided_slice_grad/StridedSliceGrad/begin:output:0:gradients/strided_slice_grad/StridedSliceGrad/end:output:0>gradients/strided_slice_grad/StridedSliceGrad/strides:output:0gradients/AddN:sum:0*
Index0*
T0*5
_output_shapes#
!:                  а*
shrink_axis_maskc
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
:«
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_11gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1+gradients_strided_slice_grad_shape_cudnnrnn%gradients_squeeze_grad_shape_cudnnrnn'gradients_squeeze_1_grad_shape_cudnnrnn6gradients/strided_slice_grad/StridedSliceGrad:output:0'gradients/Squeeze_grad/Reshape:output:0)gradients/Squeeze_1_grad/Reshape:output:0gradients_zeros_like_cudnnrnn*
T0*l
_output_shapesZ
X:                  :         а:         а:ђ╗ќ
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:┘
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*4
_output_shapes"
 :                  u
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
:к
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*(
_output_shapes
:         аy
!gradients/ExpandDims_1_grad/ShapeShape(gradients_expanddims_1_grad_shape_init_c*
T0*
_output_shapes
:╩
#gradients/ExpandDims_1_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_c_backprop:0*gradients/ExpandDims_1_grad/Shape:output:0*
T0*(
_output_shapes
:         а^
gradients/concat_1_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :џ
gradients/concat_1_grad/modFloorMod)gradients_concat_1_grad_mod_concat_1_axis%gradients/concat_1_grad/Rank:output:0*
T0*
_output_shapes
: h
gradients/concat_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:аj
gradients/concat_1_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:аj
gradients/concat_1_grad/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:аj
gradients/concat_1_grad/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:аk
gradients/concat_1_grad/Shape_4Const*
_output_shapes
:*
dtype0*
valueB:ђѕk
gradients/concat_1_grad/Shape_5Const*
_output_shapes
:*
dtype0*
valueB:ђѕk
gradients/concat_1_grad/Shape_6Const*
_output_shapes
:*
dtype0*
valueB:ђѕk
gradients/concat_1_grad/Shape_7Const*
_output_shapes
:*
dtype0*
valueB:ђѕj
gradients/concat_1_grad/Shape_8Const*
_output_shapes
:*
dtype0*
valueB:аj
gradients/concat_1_grad/Shape_9Const*
_output_shapes
:*
dtype0*
valueB:аk
 gradients/concat_1_grad/Shape_10Const*
_output_shapes
:*
dtype0*
valueB:аk
 gradients/concat_1_grad/Shape_11Const*
_output_shapes
:*
dtype0*
valueB:аk
 gradients/concat_1_grad/Shape_12Const*
_output_shapes
:*
dtype0*
valueB:аk
 gradients/concat_1_grad/Shape_13Const*
_output_shapes
:*
dtype0*
valueB:аk
 gradients/concat_1_grad/Shape_14Const*
_output_shapes
:*
dtype0*
valueB:аk
 gradients/concat_1_grad/Shape_15Const*
_output_shapes
:*
dtype0*
valueB:аЭ
$gradients/concat_1_grad/ConcatOffsetConcatOffsetgradients/concat_1_grad/mod:z:0&gradients/concat_1_grad/Shape:output:0(gradients/concat_1_grad/Shape_1:output:0(gradients/concat_1_grad/Shape_2:output:0(gradients/concat_1_grad/Shape_3:output:0(gradients/concat_1_grad/Shape_4:output:0(gradients/concat_1_grad/Shape_5:output:0(gradients/concat_1_grad/Shape_6:output:0(gradients/concat_1_grad/Shape_7:output:0(gradients/concat_1_grad/Shape_8:output:0(gradients/concat_1_grad/Shape_9:output:0)gradients/concat_1_grad/Shape_10:output:0)gradients/concat_1_grad/Shape_11:output:0)gradients/concat_1_grad/Shape_12:output:0)gradients/concat_1_grad/Shape_13:output:0)gradients/concat_1_grad/Shape_14:output:0)gradients/concat_1_grad/Shape_15:output:0*
N*t
_output_shapesb
`::::::::::::::::В
gradients/concat_1_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:0&gradients/concat_1_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:а­
gradients/concat_1_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:1(gradients/concat_1_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:а­
gradients/concat_1_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:2(gradients/concat_1_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes	
:а­
gradients/concat_1_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:3(gradients/concat_1_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes	
:аы
gradients/concat_1_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:4(gradients/concat_1_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes

:ђѕы
gradients/concat_1_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:5(gradients/concat_1_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes

:ђѕы
gradients/concat_1_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:6(gradients/concat_1_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes

:ђѕы
gradients/concat_1_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:7(gradients/concat_1_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes

:ђѕ­
gradients/concat_1_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:8(gradients/concat_1_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes	
:а­
gradients/concat_1_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:9(gradients/concat_1_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes	
:аз
 gradients/concat_1_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:10)gradients/concat_1_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes	
:аз
 gradients/concat_1_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:11)gradients/concat_1_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes	
:аз
 gradients/concat_1_grad/Slice_12Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:12)gradients/concat_1_grad/Shape_12:output:0*
Index0*
T0*
_output_shapes	
:аз
 gradients/concat_1_grad/Slice_13Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:13)gradients/concat_1_grad/Shape_13:output:0*
Index0*
T0*
_output_shapes	
:аз
 gradients/concat_1_grad/Slice_14Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:14)gradients/concat_1_grad/Shape_14:output:0*
Index0*
T0*
_output_shapes	
:аз
 gradients/concat_1_grad/Slice_15Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:15)gradients/concat_1_grad/Shape_15:output:0*
Index0*
T0*
_output_shapes	
:аm
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      б
gradients/Reshape_grad/ReshapeReshape&gradients/concat_1_grad/Slice:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes
:	аo
gradients/Reshape_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      е
 gradients/Reshape_1_grad/ReshapeReshape(gradients/concat_1_grad/Slice_1:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0*
_output_shapes
:	аo
gradients/Reshape_2_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      е
 gradients/Reshape_2_grad/ReshapeReshape(gradients/concat_1_grad/Slice_2:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0*
_output_shapes
:	аo
gradients/Reshape_3_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      е
 gradients/Reshape_3_grad/ReshapeReshape(gradients/concat_1_grad/Slice_3:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0*
_output_shapes
:	аo
gradients/Reshape_4_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      Е
 gradients/Reshape_4_grad/ReshapeReshape(gradients/concat_1_grad/Slice_4:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0* 
_output_shapes
:
ааo
gradients/Reshape_5_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      Е
 gradients/Reshape_5_grad/ReshapeReshape(gradients/concat_1_grad/Slice_5:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0* 
_output_shapes
:
ааo
gradients/Reshape_6_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      Е
 gradients/Reshape_6_grad/ReshapeReshape(gradients/concat_1_grad/Slice_6:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0* 
_output_shapes
:
ааo
gradients/Reshape_7_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      Е
 gradients/Reshape_7_grad/ReshapeReshape(gradients/concat_1_grad/Slice_7:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0* 
_output_shapes
:
ааi
gradients/Reshape_8_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:ац
 gradients/Reshape_8_grad/ReshapeReshape(gradients/concat_1_grad/Slice_8:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes	
:аi
gradients/Reshape_9_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:ац
 gradients/Reshape_9_grad/ReshapeReshape(gradients/concat_1_grad/Slice_9:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes	
:аj
gradients/Reshape_10_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:аД
!gradients/Reshape_10_grad/ReshapeReshape)gradients/concat_1_grad/Slice_10:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes	
:аj
gradients/Reshape_11_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:аД
!gradients/Reshape_11_grad/ReshapeReshape)gradients/concat_1_grad/Slice_11:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes	
:аj
gradients/Reshape_12_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:аД
!gradients/Reshape_12_grad/ReshapeReshape)gradients/concat_1_grad/Slice_12:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes	
:аj
gradients/Reshape_13_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:аД
!gradients/Reshape_13_grad/ReshapeReshape)gradients/concat_1_grad/Slice_13:output:0(gradients/Reshape_13_grad/Shape:output:0*
T0*
_output_shapes	
:аj
gradients/Reshape_14_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:аД
!gradients/Reshape_14_grad/ReshapeReshape)gradients/concat_1_grad/Slice_14:output:0(gradients/Reshape_14_grad/Shape:output:0*
T0*
_output_shapes	
:аj
gradients/Reshape_15_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:аД
!gradients/Reshape_15_grad/ReshapeReshape)gradients/concat_1_grad/Slice_15:output:0(gradients/Reshape_15_grad/Shape:output:0*
T0*
_output_shapes	
:аю
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:Х
$gradients/transpose_1_grad/transpose	Transpose'gradients/Reshape_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0*
_output_shapes
:	аю
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:И
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0*
_output_shapes
:	аю
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:И
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0*
_output_shapes
:	аю
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:И
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0*
_output_shapes
:	аю
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:╣
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
ааю
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:╣
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
ааю
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:╣
$gradients/transpose_7_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
ааю
,gradients/transpose_8_grad/InvertPermutationInvertPermutation=gradients_transpose_8_grad_invertpermutation_transpose_8_perm*
_output_shapes
:╣
$gradients/transpose_8_grad/transpose	Transpose)gradients/Reshape_7_grad/Reshape:output:00gradients/transpose_8_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
аау
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0*gradients/Reshape_13_grad/Reshape:output:0*gradients/Reshape_14_grad/Reshape:output:0*gradients/Reshape_15_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
N*
T0*
_output_shapes	
:ђ»
gradients/split_grad/concatConcatV2(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
N*
T0*
_output_shapes
:	ђ	Х
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0(gradients/transpose_7_grad/transpose:y:0(gradients/transpose_8_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
N*
T0* 
_output_shapes
:
ађ	\
gradients/concat_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :њ
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: f
gradients/concat_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:ђ	h
gradients/concat_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:ђ	╩
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0*
N* 
_output_shapes
::м
gradients/concat_grad/SliceSlice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:ђ	о
gradients/concat_grad/Slice_1Slice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:ђ	{
IdentityIdentity&gradients/transpose_grad/transpose:y:0*
T0*4
_output_shapes"
 :                  u

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0*
T0*(
_output_shapes
:         аw

Identity_2Identity,gradients/ExpandDims_1_grad/Reshape:output:0*
T0*(
_output_shapes
:         аf

Identity_3Identity$gradients/split_grad/concat:output:0*
T0*
_output_shapes
:	ђ	i

Identity_4Identity&gradients/split_1_grad/concat:output:0*
T0* 
_output_shapes
:
ађ	d

Identity_5Identity&gradients/concat_grad/Slice_1:output:0*
T0*
_output_shapes	
:ђ	"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*(
_construction_contextkEagerRuntime*ћ
_input_shapesѓ
 :         а:         а:         а:         а: :         а:         а:         а:                  а::                  :         а:         а:ђ╗::         а:         а: ::::::::: : : : *=
api_implements+)lstm_bc8f3e75-dc57-4569-96f0-8e80ed216f21*
api_preferred_deviceGPU*C
forward_function_name*(__forward_gpu_lstm_with_fallback_1212999*
go_backwards( *

time_major( :. *
(
_output_shapes
:         а:2.
,
_output_shapes
:         а:.*
(
_output_shapes
:         а:.*
(
_output_shapes
:         а:

_output_shapes
: :.*
(
_output_shapes
:         а:2.
,
_output_shapes
:         а:2.
,
_output_shapes
:         а:;7
5
_output_shapes#
!:                  а:	

_output_shapes
:::
6
4
_output_shapes"
 :                  :2.
,
_output_shapes
:         а:2.
,
_output_shapes
:         а:"

_output_shapes

:ђ╗: 

_output_shapes
::.*
(
_output_shapes
:         а:.*
(
_output_shapes
:         а:

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
╬F
»
"__inference__wrapped_model_1208164
input_1U
Bchange_layer_12_sequential_13_lstm_13_read_readvariableop_resource:	ђ	X
Dchange_layer_12_sequential_13_lstm_13_read_1_readvariableop_resource:
ађ	S
Dchange_layer_12_sequential_13_lstm_13_read_2_readvariableop_resource:	ђ	X
Echange_layer_12_sequential_13_dense_13_matmul_readvariableop_resource:	аT
Fchange_layer_12_sequential_13_dense_13_biasadd_readvariableop_resource:
identityѕб=change_layer_12/sequential_13/dense_13/BiasAdd/ReadVariableOpб<change_layer_12/sequential_13/dense_13/MatMul/ReadVariableOpб9change_layer_12/sequential_13/lstm_13/Read/ReadVariableOpб;change_layer_12/sequential_13/lstm_13/Read_1/ReadVariableOpб;change_layer_12/sequential_13/lstm_13/Read_2/ReadVariableOpb
+change_layer_12/sequential_13/lstm_13/ShapeShapeinput_1*
T0*
_output_shapes
:Ѓ
9change_layer_12/sequential_13/lstm_13/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: Ё
;change_layer_12/sequential_13/lstm_13/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Ё
;change_layer_12/sequential_13/lstm_13/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ј
3change_layer_12/sequential_13/lstm_13/strided_sliceStridedSlice4change_layer_12/sequential_13/lstm_13/Shape:output:0Bchange_layer_12/sequential_13/lstm_13/strided_slice/stack:output:0Dchange_layer_12/sequential_13/lstm_13/strided_slice/stack_1:output:0Dchange_layer_12/sequential_13/lstm_13/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskw
4change_layer_12/sequential_13/lstm_13/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :ат
2change_layer_12/sequential_13/lstm_13/zeros/packedPack<change_layer_12/sequential_13/lstm_13/strided_slice:output:0=change_layer_12/sequential_13/lstm_13/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:v
1change_layer_12/sequential_13/lstm_13/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ▀
+change_layer_12/sequential_13/lstm_13/zerosFill;change_layer_12/sequential_13/lstm_13/zeros/packed:output:0:change_layer_12/sequential_13/lstm_13/zeros/Const:output:0*
T0*(
_output_shapes
:         аy
6change_layer_12/sequential_13/lstm_13/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :аж
4change_layer_12/sequential_13/lstm_13/zeros_1/packedPack<change_layer_12/sequential_13/lstm_13/strided_slice:output:0?change_layer_12/sequential_13/lstm_13/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:x
3change_layer_12/sequential_13/lstm_13/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    т
-change_layer_12/sequential_13/lstm_13/zeros_1Fill=change_layer_12/sequential_13/lstm_13/zeros_1/packed:output:0<change_layer_12/sequential_13/lstm_13/zeros_1/Const:output:0*
T0*(
_output_shapes
:         ай
9change_layer_12/sequential_13/lstm_13/Read/ReadVariableOpReadVariableOpBchange_layer_12_sequential_13_lstm_13_read_readvariableop_resource*
_output_shapes
:	ђ	*
dtype0Д
.change_layer_12/sequential_13/lstm_13/IdentityIdentityAchange_layer_12/sequential_13/lstm_13/Read/ReadVariableOp:value:0*
T0*
_output_shapes
:	ђ	┬
;change_layer_12/sequential_13/lstm_13/Read_1/ReadVariableOpReadVariableOpDchange_layer_12_sequential_13_lstm_13_read_1_readvariableop_resource* 
_output_shapes
:
ађ	*
dtype0г
0change_layer_12/sequential_13/lstm_13/Identity_1IdentityCchange_layer_12/sequential_13/lstm_13/Read_1/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ађ	й
;change_layer_12/sequential_13/lstm_13/Read_2/ReadVariableOpReadVariableOpDchange_layer_12_sequential_13_lstm_13_read_2_readvariableop_resource*
_output_shapes	
:ђ	*
dtype0Д
0change_layer_12/sequential_13/lstm_13/Identity_2IdentityCchange_layer_12/sequential_13/lstm_13/Read_2/ReadVariableOp:value:0*
T0*
_output_shapes	
:ђ	ю
5change_layer_12/sequential_13/lstm_13/PartitionedCallPartitionedCallinput_14change_layer_12/sequential_13/lstm_13/zeros:output:06change_layer_12/sequential_13/lstm_13/zeros_1:output:07change_layer_12/sequential_13/lstm_13/Identity:output:09change_layer_12/sequential_13/lstm_13/Identity_1:output:09change_layer_12/sequential_13/lstm_13/Identity_2:output:0*
Tin

2*
Tout	
2*
_collective_manager_ids
 *j
_output_shapesX
V:         а:         а:         а:         а: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ **
f%R#
!__inference_standard_lstm_1207873├
<change_layer_12/sequential_13/dense_13/MatMul/ReadVariableOpReadVariableOpEchange_layer_12_sequential_13_dense_13_matmul_readvariableop_resource*
_output_shapes
:	а*
dtype0№
-change_layer_12/sequential_13/dense_13/MatMulMatMul>change_layer_12/sequential_13/lstm_13/PartitionedCall:output:0Dchange_layer_12/sequential_13/dense_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         └
=change_layer_12/sequential_13/dense_13/BiasAdd/ReadVariableOpReadVariableOpFchange_layer_12_sequential_13_dense_13_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0в
.change_layer_12/sequential_13/dense_13/BiasAddBiasAdd7change_layer_12/sequential_13/dense_13/MatMul:product:0Echange_layer_12/sequential_13/dense_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         Ћ
.change_layer_12/sequential_13/reshape_13/ShapeShape7change_layer_12/sequential_13/dense_13/BiasAdd:output:0*
T0*
_output_shapes
:є
<change_layer_12/sequential_13/reshape_13/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ѕ
>change_layer_12/sequential_13/reshape_13/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:ѕ
>change_layer_12/sequential_13/reshape_13/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ъ
6change_layer_12/sequential_13/reshape_13/strided_sliceStridedSlice7change_layer_12/sequential_13/reshape_13/Shape:output:0Echange_layer_12/sequential_13/reshape_13/strided_slice/stack:output:0Gchange_layer_12/sequential_13/reshape_13/strided_slice/stack_1:output:0Gchange_layer_12/sequential_13/reshape_13/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskz
8change_layer_12/sequential_13/reshape_13/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :z
8change_layer_12/sequential_13/reshape_13/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :│
6change_layer_12/sequential_13/reshape_13/Reshape/shapePack?change_layer_12/sequential_13/reshape_13/strided_slice:output:0Achange_layer_12/sequential_13/reshape_13/Reshape/shape/1:output:0Achange_layer_12/sequential_13/reshape_13/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:в
0change_layer_12/sequential_13/reshape_13/ReshapeReshape7change_layer_12/sequential_13/dense_13/BiasAdd:output:0?change_layer_12/sequential_13/reshape_13/Reshape/shape:output:0*
T0*+
_output_shapes
:         ќ
change_layer_12/addAddV2input_19change_layer_12/sequential_13/reshape_13/Reshape:output:0*
T0*+
_output_shapes
:         j
IdentityIdentitychange_layer_12/add:z:0^NoOp*
T0*+
_output_shapes
:         §
NoOpNoOp>^change_layer_12/sequential_13/dense_13/BiasAdd/ReadVariableOp=^change_layer_12/sequential_13/dense_13/MatMul/ReadVariableOp:^change_layer_12/sequential_13/lstm_13/Read/ReadVariableOp<^change_layer_12/sequential_13/lstm_13/Read_1/ReadVariableOp<^change_layer_12/sequential_13/lstm_13/Read_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:         : : : : : 2~
=change_layer_12/sequential_13/dense_13/BiasAdd/ReadVariableOp=change_layer_12/sequential_13/dense_13/BiasAdd/ReadVariableOp2|
<change_layer_12/sequential_13/dense_13/MatMul/ReadVariableOp<change_layer_12/sequential_13/dense_13/MatMul/ReadVariableOp2v
9change_layer_12/sequential_13/lstm_13/Read/ReadVariableOp9change_layer_12/sequential_13/lstm_13/Read/ReadVariableOp2z
;change_layer_12/sequential_13/lstm_13/Read_1/ReadVariableOp;change_layer_12/sequential_13/lstm_13/Read_1/ReadVariableOp2z
;change_layer_12/sequential_13/lstm_13/Read_2/ReadVariableOp;change_layer_12/sequential_13/lstm_13/Read_2/ReadVariableOp:T P
+
_output_shapes
:         
!
_user_specified_name	input_1
с
З
/__inference_sequential_13_layer_call_fn_1211204

inputs
unknown:	ђ	
	unknown_0:
ађ	
	unknown_1:	ђ	
	unknown_2:	а
	unknown_3:
identityѕбStatefulPartitionedCallі
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         *'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8ѓ *S
fNRL
J__inference_sequential_13_layer_call_and_return_conditional_losses_1210043s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:         : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
эM
б
(__forward_gpu_lstm_with_fallback_1210708

inputs
init_h_0
init_c_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4
strided_slice
cudnnrnn

cudnnrnn_0

cudnnrnn_1

cudnnrnn_2
	transpose

expanddims
expanddims_1
concat_1
transpose_perm

init_h

init_c
concat_1_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
transpose_7_perm
transpose_8_perm
split_2_split_dim
split_split_dim
split_1_split_dim
concat_axisѕc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          B
transpose_0	Transposeinputstranspose/perm:output:0*
T0P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : r

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T0*,
_output_shapes
:         аR
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : v
ExpandDims_1
ExpandDimsinit_c_0ExpandDims_1/dim:output:0*
T0*,
_output_shapes
:         аQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ї
splitSplitsplit/split_dim:output:0kernel*
T0*@
_output_shapes.
,:	а:	а:	а:	а*
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ъ
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*D
_output_shapes2
0:
аа:
аа:
аа:
аа*
	num_splite
zeros_like/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB:ђ	U
zeros_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    x

zeros_likeFill#zeros_like/shape_as_tensor:output:0zeros_like/Const:output:0*
T0*
_output_shapes	
:ђ	M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : r
concatConcatV2zeros_like:output:0biasconcat/axis:output:0*
N*
T0*
_output_shapes	
:ђS
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : Ц
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*L
_output_shapes:
8:а:а:а:а:а:а:а:а*
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
         a
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       m
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes
:	аY
ReshapeReshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:аa
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       m
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes
:	а[
	Reshape_1Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:аa
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       m
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes
:	а[
	Reshape_2Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:аa
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       m
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0*
_output_shapes
:	а[
	Reshape_3Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes	
:аa
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       p
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0* 
_output_shapes
:
аа\
	Reshape_4Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes

:ђѕa
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       p
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0* 
_output_shapes
:
аа\
	Reshape_5Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes

:ђѕa
transpose_7/permConst*
_output_shapes
:*
dtype0*
valueB"       p
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0* 
_output_shapes
:
аа\
	Reshape_6Reshapetranspose_7:y:0Const:output:0*
T0*
_output_shapes

:ђѕa
transpose_8/permConst*
_output_shapes
:*
dtype0*
valueB"       p
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0* 
_output_shapes
:
аа\
	Reshape_7Reshapetranspose_8:y:0Const:output:0*
T0*
_output_shapes

:ђѕ\
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes	
:а\
	Reshape_9Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes	
:а]

Reshape_10Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes	
:а]

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes	
:а]

Reshape_12Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes	
:а]

Reshape_13Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes	
:а]

Reshape_14Reshapesplit_2:output:6Const:output:0*
T0*
_output_shapes	
:а]

Reshape_15Reshapesplit_2:output:7Const:output:0*
T0*
_output_shapes	
:аO
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ё

concat_1_0ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T0Л
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1_0:output:0*
T0*`
_output_shapesN
L:         а:         а:         а:f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
         _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Й
strided_slice_0StridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
shrink_axis_maskq
SqueezeSqueezeCudnnRNN:output_h:0*
T0*(
_output_shapes
:         а*
squeeze_dims
 s
	Squeeze_1SqueezeCudnnRNN:output_c:0*
T0*(
_output_shapes
:         а*
squeeze_dims
 R
ExpandDims_2/dimConst*
_output_shapes
: *
dtype0*
value	B :є
ExpandDims_2
ExpandDimsstrided_slice_0:output:0ExpandDims_2/dim:output:0*
T0*,
_output_shapes
:         а[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @a
IdentityIdentitystrided_slice_0:output:0*
T0*(
_output_shapes
:         аd

Identity_1IdentityExpandDims_2:output:0*
T0*,
_output_shapes
:         а[

Identity_2IdentitySqueeze:output:0*
T0*(
_output_shapes
:         а]

Identity_3IdentitySqueeze_1:output:0*
T0*(
_output_shapes
:         аI

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: "
concat_1concat_1_0:output:0"'
concat_1_axisconcat_1/axis:output:0"#
concat_axisconcat/axis:output:0"
cudnnrnnCudnnRNN:output_h:0"!

cudnnrnn_0CudnnRNN:output_c:0"

cudnnrnn_1CudnnRNN:output:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"!

expanddimsExpandDims:output:0"%
expanddims_1ExpandDims_1:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
init_cinit_c_0"
init_hinit_h_0"/
split_1_split_dimsplit_1/split_dim:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"+
split_split_dimsplit/split_dim:output:0")
strided_slicestrided_slice_0:output:0"
	transposetranspose_0:y:0"-
transpose_1_permtranspose_1/perm:output:0"-
transpose_2_permtranspose_2/perm:output:0"-
transpose_3_permtranspose_3/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"-
transpose_5_permtranspose_5/perm:output:0"-
transpose_6_permtranspose_6/perm:output:0"-
transpose_7_permtranspose_7/perm:output:0"-
transpose_8_permtranspose_8/perm:output:0")
transpose_permtranspose/perm:output:0*(
_construction_contextkEagerRuntime*p
_input_shapes_
]:         :         а:         а:	ђ	:
ађ	:ђ	*=
api_implements+)lstm_eb22870e-927f-49a2-b687-85a1e58290d1*
api_preferred_deviceGPU*Y
backward_function_name?=__inference___backward_gpu_lstm_with_fallback_1210533_1210709*
go_backwards( *

time_major( :S O
+
_output_shapes
:         
 
_user_specified_nameinputs:PL
(
_output_shapes
:         а
 
_user_specified_nameinit_h:PL
(
_output_shapes
:         а
 
_user_specified_nameinit_c:GC

_output_shapes
:	ђ	
 
_user_specified_namekernel:RN
 
_output_shapes
:
ађ	
*
_user_specified_namerecurrent_kernel:A=

_output_shapes	
:ђ	

_user_specified_namebias
ћ
╣
)__inference_lstm_13_layer_call_fn_1212107
inputs_0
unknown:	ђ	
	unknown_0:
ађ	
	unknown_1:	ђ	
identityѕбStatefulPartitionedCallж
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         а*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_lstm_13_layer_call_and_return_conditional_losses_1208599p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         а`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :                  
"
_user_specified_name
inputs_0
с
З
/__inference_sequential_13_layer_call_fn_1211189

inputs
unknown:	ђ	
	unknown_0:
ађ	
	unknown_1:	ђ	
	unknown_2:	а
	unknown_3:
identityѕбStatefulPartitionedCallі
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         *'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8ѓ *S
fNRL
J__inference_sequential_13_layer_call_and_return_conditional_losses_1209527s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:         : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
х
┐
D__inference_lstm_13_layer_call_and_return_conditional_losses_1209041

inputs/
read_readvariableop_resource:	ђ	2
read_1_readvariableop_resource:
ађ	-
read_2_readvariableop_resource:	ђ	

identity_3ѕбRead/ReadVariableOpбRead_1/ReadVariableOpбRead_2/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Л
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :аs
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:         аS
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :аw
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    s
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:         аq
Read/ReadVariableOpReadVariableOpread_readvariableop_resource*
_output_shapes
:	ђ	*
dtype0[
IdentityIdentityRead/ReadVariableOp:value:0*
T0*
_output_shapes
:	ђ	v
Read_1/ReadVariableOpReadVariableOpread_1_readvariableop_resource* 
_output_shapes
:
ађ	*
dtype0`

Identity_1IdentityRead_1/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ађ	q
Read_2/ReadVariableOpReadVariableOpread_2_readvariableop_resource*
_output_shapes	
:ђ	*
dtype0[

Identity_2IdentityRead_2/ReadVariableOp:value:0*
T0*
_output_shapes	
:ђ	и
PartitionedCallPartitionedCallinputszeros:output:0zeros_1:output:0Identity:output:0Identity_1:output:0Identity_2:output:0*
Tin

2*
Tout	
2*
_collective_manager_ids
 *j
_output_shapesX
V:         а:         а:         а:         а: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ **
f%R#
!__inference_standard_lstm_1208766j

Identity_3IdentityPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         аї
NoOpNoOp^Read/ReadVariableOp^Read_1/ReadVariableOp^Read_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  : : : 2*
Read/ReadVariableOpRead/ReadVariableOp2.
Read_1/ReadVariableOpRead_1/ReadVariableOp2.
Read_2/ReadVariableOpRead_2/ReadVariableOp:\ X
4
_output_shapes"
 :                  
 
_user_specified_nameinputs
џ	
к
while_cond_1210349
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice5
1while_while_cond_1210349___redundant_placeholder05
1while_while_cond_1210349___redundant_placeholder15
1while_while_cond_1210349___redundant_placeholder25
1while_while_cond_1210349___redundant_placeholder3
while_identity
`

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :         а:         а: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:         а:.*
(
_output_shapes
:         а:

_output_shapes
: :

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
р;
┴
!__inference_standard_lstm_1212296

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :                  B
ShapeShapetranspose:y:0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Л
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ▓
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмє
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       Я
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУм_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ж
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maske
MatMulMatMulstrided_slice_1:output:0kernel*
T0*(
_output_shapes
:         ђ	_
MatMul_1MatMulinit_hrecurrent_kernel*
T0*(
_output_shapes
:         ђ	e
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         ђ	T
BiasAddBiasAddadd:z:0bias*
T0*(
_output_shapes
:         ђ	Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :║
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:         а:         а:         а:         а*
	num_splitU
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:         аW
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:         аT
mulMulSigmoid_1:y:0init_c*
T0*(
_output_shapes
:         аO
TanhTanhsplit:output:2*
T0*(
_output_shapes
:         аV
mul_1MulSigmoid:y:0Tanh:y:0*
T0*(
_output_shapes
:         аU
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:         аW
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:         аL
Tanh_1Tanh	add_1:z:0*
T0*(
_output_shapes
:         аZ
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*(
_output_shapes
:         аn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :┼
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Й
whileStatelessWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hinit_cstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelrecurrent_kernelbias*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*f
_output_shapesT
R: : : : :         а:         а: : :	ђ	:
ађ	:ђ	* 
_read_only_resource_inputs
 *
_stateful_parallelism( *
bodyR
while_body_1212210*
condR
while_cond_1212209*e
output_shapesT
R: : : : :         а:         а: : :	ђ	:
ађ	:ђ	*
parallel_iterations Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       О
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:         а*
element_dtype0*
num_elementsh
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ѕ
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         а*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ќ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:         а[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  ђ?a
IdentityIdentitystrided_slice_2:output:0*
T0*(
_output_shapes
:         а^

Identity_1Identitytranspose_1:y:0*
T0*,
_output_shapes
:         аY

Identity_2Identitywhile:output:4*
T0*(
_output_shapes
:         аY

Identity_3Identitywhile:output:5*
T0*(
_output_shapes
:         аI

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*y
_input_shapesh
f:                  :         а:         а:	ђ	:
ађ	:ђ	*=
api_implements+)lstm_8e4ab49d-552b-4e19-8094-3b0af108a7d4*
api_preferred_deviceCPU*
go_backwards( *

time_major( :\ X
4
_output_shapes"
 :                  
 
_user_specified_nameinputs:PL
(
_output_shapes
:         а
 
_user_specified_nameinit_h:PL
(
_output_shapes
:         а
 
_user_specified_nameinit_c:GC

_output_shapes
:	ђ	
 
_user_specified_namekernel:RN
 
_output_shapes
:
ађ	
*
_user_specified_namerecurrent_kernel:A=

_output_shapes	
:ђ	

_user_specified_namebias
р;
┴
!__inference_standard_lstm_1212727

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :                  B
ShapeShapetranspose:y:0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Л
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ▓
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмє
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       Я
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУм_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ж
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maske
MatMulMatMulstrided_slice_1:output:0kernel*
T0*(
_output_shapes
:         ђ	_
MatMul_1MatMulinit_hrecurrent_kernel*
T0*(
_output_shapes
:         ђ	e
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         ђ	T
BiasAddBiasAddadd:z:0bias*
T0*(
_output_shapes
:         ђ	Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :║
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:         а:         а:         а:         а*
	num_splitU
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:         аW
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:         аT
mulMulSigmoid_1:y:0init_c*
T0*(
_output_shapes
:         аO
TanhTanhsplit:output:2*
T0*(
_output_shapes
:         аV
mul_1MulSigmoid:y:0Tanh:y:0*
T0*(
_output_shapes
:         аU
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:         аW
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:         аL
Tanh_1Tanh	add_1:z:0*
T0*(
_output_shapes
:         аZ
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*(
_output_shapes
:         аn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :┼
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Й
whileStatelessWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hinit_cstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelrecurrent_kernelbias*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*f
_output_shapesT
R: : : : :         а:         а: : :	ђ	:
ађ	:ђ	* 
_read_only_resource_inputs
 *
_stateful_parallelism( *
bodyR
while_body_1212641*
condR
while_cond_1212640*e
output_shapesT
R: : : : :         а:         а: : :	ђ	:
ађ	:ђ	*
parallel_iterations Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       О
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:         а*
element_dtype0*
num_elementsh
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ѕ
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         а*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ќ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:         а[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  ђ?a
IdentityIdentitystrided_slice_2:output:0*
T0*(
_output_shapes
:         а^

Identity_1Identitytranspose_1:y:0*
T0*,
_output_shapes
:         аY

Identity_2Identitywhile:output:4*
T0*(
_output_shapes
:         аY

Identity_3Identitywhile:output:5*
T0*(
_output_shapes
:         аI

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*y
_input_shapesh
f:                  :         а:         а:	ђ	:
ађ	:ђ	*=
api_implements+)lstm_bc8f3e75-dc57-4569-96f0-8e80ed216f21*
api_preferred_deviceCPU*
go_backwards( *

time_major( :\ X
4
_output_shapes"
 :                  
 
_user_specified_nameinputs:PL
(
_output_shapes
:         а
 
_user_specified_nameinit_h:PL
(
_output_shapes
:         а
 
_user_specified_nameinit_c:GC

_output_shapes
:	ђ	
 
_user_specified_namekernel:RN
 
_output_shapes
:
ађ	
*
_user_specified_namerecurrent_kernel:A=

_output_shapes	
:ђ	

_user_specified_namebias
эM
б
(__forward_gpu_lstm_with_fallback_1211155

inputs
init_h_0
init_c_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4
strided_slice
cudnnrnn

cudnnrnn_0

cudnnrnn_1

cudnnrnn_2
	transpose

expanddims
expanddims_1
concat_1
transpose_perm

init_h

init_c
concat_1_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
transpose_7_perm
transpose_8_perm
split_2_split_dim
split_split_dim
split_1_split_dim
concat_axisѕc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          B
transpose_0	Transposeinputstranspose/perm:output:0*
T0P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : r

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T0*,
_output_shapes
:         аR
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : v
ExpandDims_1
ExpandDimsinit_c_0ExpandDims_1/dim:output:0*
T0*,
_output_shapes
:         аQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ї
splitSplitsplit/split_dim:output:0kernel*
T0*@
_output_shapes.
,:	а:	а:	а:	а*
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ъ
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*D
_output_shapes2
0:
аа:
аа:
аа:
аа*
	num_splite
zeros_like/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB:ђ	U
zeros_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    x

zeros_likeFill#zeros_like/shape_as_tensor:output:0zeros_like/Const:output:0*
T0*
_output_shapes	
:ђ	M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : r
concatConcatV2zeros_like:output:0biasconcat/axis:output:0*
N*
T0*
_output_shapes	
:ђS
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : Ц
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*L
_output_shapes:
8:а:а:а:а:а:а:а:а*
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
         a
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       m
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes
:	аY
ReshapeReshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:аa
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       m
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes
:	а[
	Reshape_1Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:аa
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       m
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes
:	а[
	Reshape_2Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:аa
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       m
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0*
_output_shapes
:	а[
	Reshape_3Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes	
:аa
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       p
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0* 
_output_shapes
:
аа\
	Reshape_4Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes

:ђѕa
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       p
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0* 
_output_shapes
:
аа\
	Reshape_5Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes

:ђѕa
transpose_7/permConst*
_output_shapes
:*
dtype0*
valueB"       p
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0* 
_output_shapes
:
аа\
	Reshape_6Reshapetranspose_7:y:0Const:output:0*
T0*
_output_shapes

:ђѕa
transpose_8/permConst*
_output_shapes
:*
dtype0*
valueB"       p
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0* 
_output_shapes
:
аа\
	Reshape_7Reshapetranspose_8:y:0Const:output:0*
T0*
_output_shapes

:ђѕ\
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes	
:а\
	Reshape_9Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes	
:а]

Reshape_10Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes	
:а]

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes	
:а]

Reshape_12Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes	
:а]

Reshape_13Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes	
:а]

Reshape_14Reshapesplit_2:output:6Const:output:0*
T0*
_output_shapes	
:а]

Reshape_15Reshapesplit_2:output:7Const:output:0*
T0*
_output_shapes	
:аO
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ё

concat_1_0ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T0Л
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1_0:output:0*
T0*`
_output_shapesN
L:         а:         а:         а:f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
         _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Й
strided_slice_0StridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
shrink_axis_maskq
SqueezeSqueezeCudnnRNN:output_h:0*
T0*(
_output_shapes
:         а*
squeeze_dims
 s
	Squeeze_1SqueezeCudnnRNN:output_c:0*
T0*(
_output_shapes
:         а*
squeeze_dims
 R
ExpandDims_2/dimConst*
_output_shapes
: *
dtype0*
value	B :є
ExpandDims_2
ExpandDimsstrided_slice_0:output:0ExpandDims_2/dim:output:0*
T0*,
_output_shapes
:         а[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @a
IdentityIdentitystrided_slice_0:output:0*
T0*(
_output_shapes
:         аd

Identity_1IdentityExpandDims_2:output:0*
T0*,
_output_shapes
:         а[

Identity_2IdentitySqueeze:output:0*
T0*(
_output_shapes
:         а]

Identity_3IdentitySqueeze_1:output:0*
T0*(
_output_shapes
:         аI

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: "
concat_1concat_1_0:output:0"'
concat_1_axisconcat_1/axis:output:0"#
concat_axisconcat/axis:output:0"
cudnnrnnCudnnRNN:output_h:0"!

cudnnrnn_0CudnnRNN:output_c:0"

cudnnrnn_1CudnnRNN:output:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"!

expanddimsExpandDims:output:0"%
expanddims_1ExpandDims_1:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
init_cinit_c_0"
init_hinit_h_0"/
split_1_split_dimsplit_1/split_dim:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"+
split_split_dimsplit/split_dim:output:0")
strided_slicestrided_slice_0:output:0"
	transposetranspose_0:y:0"-
transpose_1_permtranspose_1/perm:output:0"-
transpose_2_permtranspose_2/perm:output:0"-
transpose_3_permtranspose_3/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"-
transpose_5_permtranspose_5/perm:output:0"-
transpose_6_permtranspose_6/perm:output:0"-
transpose_7_permtranspose_7/perm:output:0"-
transpose_8_permtranspose_8/perm:output:0")
transpose_permtranspose/perm:output:0*(
_construction_contextkEagerRuntime*p
_input_shapes_
]:         :         а:         а:	ђ	:
ађ	:ђ	*=
api_implements+)lstm_55c05eb7-11a9-4616-8925-c9051a66537d*
api_preferred_deviceGPU*Y
backward_function_name?=__inference___backward_gpu_lstm_with_fallback_1210980_1211156*
go_backwards( *

time_major( :S O
+
_output_shapes
:         
 
_user_specified_nameinputs:PL
(
_output_shapes
:         а
 
_user_specified_nameinit_h:PL
(
_output_shapes
:         а
 
_user_specified_nameinit_c:GC

_output_shapes
:	ђ	
 
_user_specified_namekernel:RN
 
_output_shapes
:
ађ	
*
_user_specified_namerecurrent_kernel:A=

_output_shapes	
:ђ	

_user_specified_namebias
ф┬
у
=__inference___backward_gpu_lstm_with_fallback_1213255_1213431
placeholder
placeholder_1
placeholder_2
placeholder_3
placeholder_43
/gradients_expanddims_2_grad_shape_strided_slice)
%gradients_squeeze_grad_shape_cudnnrnn+
'gradients_squeeze_1_grad_shape_cudnnrnn/
+gradients_strided_slice_grad_shape_cudnnrnn!
gradients_zeros_like_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9
5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_15
1gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h,
(gradients_expanddims_1_grad_shape_init_c-
)gradients_concat_1_grad_mod_concat_1_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_permA
=gradients_transpose_7_grad_invertpermutation_transpose_7_permA
=gradients_transpose_8_grad_invertpermutation_transpose_8_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim)
%gradients_concat_grad_mod_concat_axis
identity

identity_1

identity_2

identity_3

identity_4

identity_5ѕ_
gradients/grad_ys_0Identityplaceholder*
T0*(
_output_shapes
:         аe
gradients/grad_ys_1Identityplaceholder_1*
T0*,
_output_shapes
:         аa
gradients/grad_ys_2Identityplaceholder_2*
T0*(
_output_shapes
:         аa
gradients/grad_ys_3Identityplaceholder_3*
T0*(
_output_shapes
:         аO
gradients/grad_ys_4Identityplaceholder_4*
T0*
_output_shapes
: ђ
!gradients/ExpandDims_2_grad/ShapeShape/gradients_expanddims_2_grad_shape_strided_slice*
T0*
_output_shapes
:Ф
#gradients/ExpandDims_2_grad/ReshapeReshapegradients/grad_ys_1:output:0*gradients/ExpandDims_2_grad/Shape:output:0*
T0*(
_output_shapes
:         аq
gradients/Squeeze_grad/ShapeShape%gradients_squeeze_grad_shape_cudnnrnn*
T0*
_output_shapes
:Ц
gradients/Squeeze_grad/ReshapeReshapegradients/grad_ys_2:output:0%gradients/Squeeze_grad/Shape:output:0*
T0*,
_output_shapes
:         аu
gradients/Squeeze_1_grad/ShapeShape'gradients_squeeze_1_grad_shape_cudnnrnn*
T0*
_output_shapes
:Е
 gradients/Squeeze_1_grad/ReshapeReshapegradients/grad_ys_3:output:0'gradients/Squeeze_1_grad/Shape:output:0*
T0*,
_output_shapes
:         ак
gradients/AddNAddNgradients/grad_ys_0:output:0,gradients/ExpandDims_2_grad/Reshape:output:0*
N*
T0*&
_class
loc:@gradients/grad_ys_0*(
_output_shapes
:         а}
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
:є
3gradients/strided_slice_grad/StridedSliceGrad/beginConst*
_output_shapes
:*
dtype0*
valueB:
         {
1gradients/strided_slice_grad/StridedSliceGrad/endConst*
_output_shapes
:*
dtype0*
valueB: 
5gradients/strided_slice_grad/StridedSliceGrad/stridesConst*
_output_shapes
:*
dtype0*
valueB:џ
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0<gradients/strided_slice_grad/StridedSliceGrad/begin:output:0:gradients/strided_slice_grad/StridedSliceGrad/end:output:0>gradients/strided_slice_grad/StridedSliceGrad/strides:output:0gradients/AddN:sum:0*
Index0*
T0*,
_output_shapes
:         а*
shrink_axis_maskc
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
:Ц
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_11gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1+gradients_strided_slice_grad_shape_cudnnrnn%gradients_squeeze_grad_shape_cudnnrnn'gradients_squeeze_1_grad_shape_cudnnrnn6gradients/strided_slice_grad/StridedSliceGrad:output:0'gradients/Squeeze_grad/Reshape:output:0)gradients/Squeeze_1_grad/Reshape:output:0gradients_zeros_like_cudnnrnn*
T0*c
_output_shapesQ
O:         :         а:         а:ђ╗ќ
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:л
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*+
_output_shapes
:         u
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
:к
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*(
_output_shapes
:         аy
!gradients/ExpandDims_1_grad/ShapeShape(gradients_expanddims_1_grad_shape_init_c*
T0*
_output_shapes
:╩
#gradients/ExpandDims_1_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_c_backprop:0*gradients/ExpandDims_1_grad/Shape:output:0*
T0*(
_output_shapes
:         а^
gradients/concat_1_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :џ
gradients/concat_1_grad/modFloorMod)gradients_concat_1_grad_mod_concat_1_axis%gradients/concat_1_grad/Rank:output:0*
T0*
_output_shapes
: h
gradients/concat_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:аj
gradients/concat_1_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:аj
gradients/concat_1_grad/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:аj
gradients/concat_1_grad/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:аk
gradients/concat_1_grad/Shape_4Const*
_output_shapes
:*
dtype0*
valueB:ђѕk
gradients/concat_1_grad/Shape_5Const*
_output_shapes
:*
dtype0*
valueB:ђѕk
gradients/concat_1_grad/Shape_6Const*
_output_shapes
:*
dtype0*
valueB:ђѕk
gradients/concat_1_grad/Shape_7Const*
_output_shapes
:*
dtype0*
valueB:ђѕj
gradients/concat_1_grad/Shape_8Const*
_output_shapes
:*
dtype0*
valueB:аj
gradients/concat_1_grad/Shape_9Const*
_output_shapes
:*
dtype0*
valueB:аk
 gradients/concat_1_grad/Shape_10Const*
_output_shapes
:*
dtype0*
valueB:аk
 gradients/concat_1_grad/Shape_11Const*
_output_shapes
:*
dtype0*
valueB:аk
 gradients/concat_1_grad/Shape_12Const*
_output_shapes
:*
dtype0*
valueB:аk
 gradients/concat_1_grad/Shape_13Const*
_output_shapes
:*
dtype0*
valueB:аk
 gradients/concat_1_grad/Shape_14Const*
_output_shapes
:*
dtype0*
valueB:аk
 gradients/concat_1_grad/Shape_15Const*
_output_shapes
:*
dtype0*
valueB:аЭ
$gradients/concat_1_grad/ConcatOffsetConcatOffsetgradients/concat_1_grad/mod:z:0&gradients/concat_1_grad/Shape:output:0(gradients/concat_1_grad/Shape_1:output:0(gradients/concat_1_grad/Shape_2:output:0(gradients/concat_1_grad/Shape_3:output:0(gradients/concat_1_grad/Shape_4:output:0(gradients/concat_1_grad/Shape_5:output:0(gradients/concat_1_grad/Shape_6:output:0(gradients/concat_1_grad/Shape_7:output:0(gradients/concat_1_grad/Shape_8:output:0(gradients/concat_1_grad/Shape_9:output:0)gradients/concat_1_grad/Shape_10:output:0)gradients/concat_1_grad/Shape_11:output:0)gradients/concat_1_grad/Shape_12:output:0)gradients/concat_1_grad/Shape_13:output:0)gradients/concat_1_grad/Shape_14:output:0)gradients/concat_1_grad/Shape_15:output:0*
N*t
_output_shapesb
`::::::::::::::::В
gradients/concat_1_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:0&gradients/concat_1_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:а­
gradients/concat_1_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:1(gradients/concat_1_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:а­
gradients/concat_1_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:2(gradients/concat_1_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes	
:а­
gradients/concat_1_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:3(gradients/concat_1_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes	
:аы
gradients/concat_1_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:4(gradients/concat_1_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes

:ђѕы
gradients/concat_1_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:5(gradients/concat_1_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes

:ђѕы
gradients/concat_1_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:6(gradients/concat_1_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes

:ђѕы
gradients/concat_1_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:7(gradients/concat_1_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes

:ђѕ­
gradients/concat_1_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:8(gradients/concat_1_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes	
:а­
gradients/concat_1_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:9(gradients/concat_1_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes	
:аз
 gradients/concat_1_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:10)gradients/concat_1_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes	
:аз
 gradients/concat_1_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:11)gradients/concat_1_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes	
:аз
 gradients/concat_1_grad/Slice_12Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:12)gradients/concat_1_grad/Shape_12:output:0*
Index0*
T0*
_output_shapes	
:аз
 gradients/concat_1_grad/Slice_13Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:13)gradients/concat_1_grad/Shape_13:output:0*
Index0*
T0*
_output_shapes	
:аз
 gradients/concat_1_grad/Slice_14Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:14)gradients/concat_1_grad/Shape_14:output:0*
Index0*
T0*
_output_shapes	
:аз
 gradients/concat_1_grad/Slice_15Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:15)gradients/concat_1_grad/Shape_15:output:0*
Index0*
T0*
_output_shapes	
:аm
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      б
gradients/Reshape_grad/ReshapeReshape&gradients/concat_1_grad/Slice:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes
:	аo
gradients/Reshape_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      е
 gradients/Reshape_1_grad/ReshapeReshape(gradients/concat_1_grad/Slice_1:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0*
_output_shapes
:	аo
gradients/Reshape_2_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      е
 gradients/Reshape_2_grad/ReshapeReshape(gradients/concat_1_grad/Slice_2:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0*
_output_shapes
:	аo
gradients/Reshape_3_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      е
 gradients/Reshape_3_grad/ReshapeReshape(gradients/concat_1_grad/Slice_3:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0*
_output_shapes
:	аo
gradients/Reshape_4_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      Е
 gradients/Reshape_4_grad/ReshapeReshape(gradients/concat_1_grad/Slice_4:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0* 
_output_shapes
:
ааo
gradients/Reshape_5_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      Е
 gradients/Reshape_5_grad/ReshapeReshape(gradients/concat_1_grad/Slice_5:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0* 
_output_shapes
:
ааo
gradients/Reshape_6_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      Е
 gradients/Reshape_6_grad/ReshapeReshape(gradients/concat_1_grad/Slice_6:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0* 
_output_shapes
:
ааo
gradients/Reshape_7_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      Е
 gradients/Reshape_7_grad/ReshapeReshape(gradients/concat_1_grad/Slice_7:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0* 
_output_shapes
:
ааi
gradients/Reshape_8_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:ац
 gradients/Reshape_8_grad/ReshapeReshape(gradients/concat_1_grad/Slice_8:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes	
:аi
gradients/Reshape_9_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:ац
 gradients/Reshape_9_grad/ReshapeReshape(gradients/concat_1_grad/Slice_9:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes	
:аj
gradients/Reshape_10_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:аД
!gradients/Reshape_10_grad/ReshapeReshape)gradients/concat_1_grad/Slice_10:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes	
:аj
gradients/Reshape_11_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:аД
!gradients/Reshape_11_grad/ReshapeReshape)gradients/concat_1_grad/Slice_11:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes	
:аj
gradients/Reshape_12_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:аД
!gradients/Reshape_12_grad/ReshapeReshape)gradients/concat_1_grad/Slice_12:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes	
:аj
gradients/Reshape_13_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:аД
!gradients/Reshape_13_grad/ReshapeReshape)gradients/concat_1_grad/Slice_13:output:0(gradients/Reshape_13_grad/Shape:output:0*
T0*
_output_shapes	
:аj
gradients/Reshape_14_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:аД
!gradients/Reshape_14_grad/ReshapeReshape)gradients/concat_1_grad/Slice_14:output:0(gradients/Reshape_14_grad/Shape:output:0*
T0*
_output_shapes	
:аj
gradients/Reshape_15_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:аД
!gradients/Reshape_15_grad/ReshapeReshape)gradients/concat_1_grad/Slice_15:output:0(gradients/Reshape_15_grad/Shape:output:0*
T0*
_output_shapes	
:аю
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:Х
$gradients/transpose_1_grad/transpose	Transpose'gradients/Reshape_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0*
_output_shapes
:	аю
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:И
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0*
_output_shapes
:	аю
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:И
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0*
_output_shapes
:	аю
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:И
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0*
_output_shapes
:	аю
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:╣
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
ааю
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:╣
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
ааю
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:╣
$gradients/transpose_7_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
ааю
,gradients/transpose_8_grad/InvertPermutationInvertPermutation=gradients_transpose_8_grad_invertpermutation_transpose_8_perm*
_output_shapes
:╣
$gradients/transpose_8_grad/transpose	Transpose)gradients/Reshape_7_grad/Reshape:output:00gradients/transpose_8_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
аау
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0*gradients/Reshape_13_grad/Reshape:output:0*gradients/Reshape_14_grad/Reshape:output:0*gradients/Reshape_15_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
N*
T0*
_output_shapes	
:ђ»
gradients/split_grad/concatConcatV2(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
N*
T0*
_output_shapes
:	ђ	Х
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0(gradients/transpose_7_grad/transpose:y:0(gradients/transpose_8_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
N*
T0* 
_output_shapes
:
ађ	\
gradients/concat_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :њ
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: f
gradients/concat_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:ђ	h
gradients/concat_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:ђ	╩
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0*
N* 
_output_shapes
::м
gradients/concat_grad/SliceSlice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:ђ	о
gradients/concat_grad/Slice_1Slice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:ђ	r
IdentityIdentity&gradients/transpose_grad/transpose:y:0*
T0*+
_output_shapes
:         u

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0*
T0*(
_output_shapes
:         аw

Identity_2Identity,gradients/ExpandDims_1_grad/Reshape:output:0*
T0*(
_output_shapes
:         аf

Identity_3Identity$gradients/split_grad/concat:output:0*
T0*
_output_shapes
:	ђ	i

Identity_4Identity&gradients/split_1_grad/concat:output:0*
T0* 
_output_shapes
:
ађ	d

Identity_5Identity&gradients/concat_grad/Slice_1:output:0*
T0*
_output_shapes	
:ђ	"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*(
_construction_contextkEagerRuntime*ѓ
_input_shapes­
ь:         а:         а:         а:         а: :         а:         а:         а:         а::         :         а:         а:ђ╗::         а:         а: ::::::::: : : : *=
api_implements+)lstm_27cab513-e281-4aef-8d50-ee58fd8ba0ea*
api_preferred_deviceGPU*C
forward_function_name*(__forward_gpu_lstm_with_fallback_1213430*
go_backwards( *

time_major( :. *
(
_output_shapes
:         а:2.
,
_output_shapes
:         а:.*
(
_output_shapes
:         а:.*
(
_output_shapes
:         а:

_output_shapes
: :.*
(
_output_shapes
:         а:2.
,
_output_shapes
:         а:2.
,
_output_shapes
:         а:2.
,
_output_shapes
:         а:	

_output_shapes
::1
-
+
_output_shapes
:         :2.
,
_output_shapes
:         а:2.
,
_output_shapes
:         а:"

_output_shapes

:ђ╗: 

_output_shapes
::.*
(
_output_shapes
:         а:.*
(
_output_shapes
:         а:

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Ц
П
L__inference_change_layer_12_layer_call_and_return_conditional_losses_1210171

inputs(
sequential_13_1210158:	ђ	)
sequential_13_1210160:
ађ	$
sequential_13_1210162:	ђ	(
sequential_13_1210164:	а#
sequential_13_1210166:
identityѕб%sequential_13/StatefulPartitionedCallо
%sequential_13/StatefulPartitionedCallStatefulPartitionedCallinputssequential_13_1210158sequential_13_1210160sequential_13_1210162sequential_13_1210164sequential_13_1210166*
Tin

2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         *'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8ѓ *S
fNRL
J__inference_sequential_13_layer_call_and_return_conditional_losses_1210043z
addAddV2inputs.sequential_13/StatefulPartitionedCall:output:0*
T0*+
_output_shapes
:         Z
IdentityIdentityadd:z:0^NoOp*
T0*+
_output_shapes
:         n
NoOpNoOp&^sequential_13/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:         : : : : : 2N
%sequential_13/StatefulPartitionedCall%sequential_13/StatefulPartitionedCall:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
к;
┴
!__inference_standard_lstm_1213589

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:         B
ShapeShapetranspose:y:0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Л
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ▓
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмє
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       Я
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУм_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ж
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maske
MatMulMatMulstrided_slice_1:output:0kernel*
T0*(
_output_shapes
:         ђ	_
MatMul_1MatMulinit_hrecurrent_kernel*
T0*(
_output_shapes
:         ђ	e
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         ђ	T
BiasAddBiasAddadd:z:0bias*
T0*(
_output_shapes
:         ђ	Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :║
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:         а:         а:         а:         а*
	num_splitU
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:         аW
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:         аT
mulMulSigmoid_1:y:0init_c*
T0*(
_output_shapes
:         аO
TanhTanhsplit:output:2*
T0*(
_output_shapes
:         аV
mul_1MulSigmoid:y:0Tanh:y:0*
T0*(
_output_shapes
:         аU
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:         аW
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:         аL
Tanh_1Tanh	add_1:z:0*
T0*(
_output_shapes
:         аZ
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*(
_output_shapes
:         аn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :┼
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Й
whileStatelessWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hinit_cstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelrecurrent_kernelbias*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*f
_output_shapesT
R: : : : :         а:         а: : :	ђ	:
ађ	:ђ	* 
_read_only_resource_inputs
 *
_stateful_parallelism( *
bodyR
while_body_1213503*
condR
while_cond_1213502*e
output_shapesT
R: : : : :         а:         а: : :	ђ	:
ађ	:ђ	*
parallel_iterations Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       О
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:         а*
element_dtype0*
num_elementsh
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ѕ
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         а*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ќ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:         а[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  ђ?a
IdentityIdentitystrided_slice_2:output:0*
T0*(
_output_shapes
:         а^

Identity_1Identitytranspose_1:y:0*
T0*,
_output_shapes
:         аY

Identity_2Identitywhile:output:4*
T0*(
_output_shapes
:         аY

Identity_3Identitywhile:output:5*
T0*(
_output_shapes
:         аI

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*p
_input_shapes_
]:         :         а:         а:	ђ	:
ађ	:ђ	*=
api_implements+)lstm_1013b0ea-424a-45a4-86e4-0f580df52117*
api_preferred_deviceCPU*
go_backwards( *

time_major( :S O
+
_output_shapes
:         
 
_user_specified_nameinputs:PL
(
_output_shapes
:         а
 
_user_specified_nameinit_h:PL
(
_output_shapes
:         а
 
_user_specified_nameinit_c:GC

_output_shapes
:	ђ	
 
_user_specified_namekernel:RN
 
_output_shapes
:
ађ	
*
_user_specified_namerecurrent_kernel:A=

_output_shapes	
:ђ	

_user_specified_namebias
џ	
к
while_cond_1209125
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice5
1while_while_cond_1209125___redundant_placeholder05
1while_while_cond_1209125___redundant_placeholder15
1while_while_cond_1209125___redundant_placeholder25
1while_while_cond_1209125___redundant_placeholder3
while_identity
`

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :         а:         а: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:         а:.*
(
_output_shapes
:         а:

_output_shapes
: :

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
к;
┴
!__inference_standard_lstm_1211806

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:         B
ShapeShapetranspose:y:0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Л
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ▓
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмє
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       Я
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУм_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ж
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maske
MatMulMatMulstrided_slice_1:output:0kernel*
T0*(
_output_shapes
:         ђ	_
MatMul_1MatMulinit_hrecurrent_kernel*
T0*(
_output_shapes
:         ђ	e
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         ђ	T
BiasAddBiasAddadd:z:0bias*
T0*(
_output_shapes
:         ђ	Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :║
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:         а:         а:         а:         а*
	num_splitU
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:         аW
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:         аT
mulMulSigmoid_1:y:0init_c*
T0*(
_output_shapes
:         аO
TanhTanhsplit:output:2*
T0*(
_output_shapes
:         аV
mul_1MulSigmoid:y:0Tanh:y:0*
T0*(
_output_shapes
:         аU
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:         аW
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:         аL
Tanh_1Tanh	add_1:z:0*
T0*(
_output_shapes
:         аZ
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*(
_output_shapes
:         аn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :┼
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Й
whileStatelessWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hinit_cstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelrecurrent_kernelbias*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*f
_output_shapesT
R: : : : :         а:         а: : :	ђ	:
ађ	:ђ	* 
_read_only_resource_inputs
 *
_stateful_parallelism( *
bodyR
while_body_1211720*
condR
while_cond_1211719*e
output_shapesT
R: : : : :         а:         а: : :	ђ	:
ађ	:ђ	*
parallel_iterations Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       О
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:         а*
element_dtype0*
num_elementsh
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ѕ
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         а*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ќ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:         а[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  ђ?a
IdentityIdentitystrided_slice_2:output:0*
T0*(
_output_shapes
:         а^

Identity_1Identitytranspose_1:y:0*
T0*,
_output_shapes
:         аY

Identity_2Identitywhile:output:4*
T0*(
_output_shapes
:         аY

Identity_3Identitywhile:output:5*
T0*(
_output_shapes
:         аI

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*p
_input_shapes_
]:         :         а:         а:	ђ	:
ађ	:ђ	*=
api_implements+)lstm_f0644406-b4c4-48ed-b733-8b83744a8936*
api_preferred_deviceCPU*
go_backwards( *

time_major( :S O
+
_output_shapes
:         
 
_user_specified_nameinputs:PL
(
_output_shapes
:         а
 
_user_specified_nameinit_h:PL
(
_output_shapes
:         а
 
_user_specified_nameinit_c:GC

_output_shapes
:	ђ	
 
_user_specified_namekernel:RN
 
_output_shapes
:
ађ	
*
_user_specified_namerecurrent_kernel:A=

_output_shapes	
:ђ	

_user_specified_namebias
к;
┴
!__inference_standard_lstm_1210436

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:         B
ShapeShapetranspose:y:0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Л
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ▓
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмє
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       Я
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУм_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ж
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maske
MatMulMatMulstrided_slice_1:output:0kernel*
T0*(
_output_shapes
:         ђ	_
MatMul_1MatMulinit_hrecurrent_kernel*
T0*(
_output_shapes
:         ђ	e
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         ђ	T
BiasAddBiasAddadd:z:0bias*
T0*(
_output_shapes
:         ђ	Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :║
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:         а:         а:         а:         а*
	num_splitU
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:         аW
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:         аT
mulMulSigmoid_1:y:0init_c*
T0*(
_output_shapes
:         аO
TanhTanhsplit:output:2*
T0*(
_output_shapes
:         аV
mul_1MulSigmoid:y:0Tanh:y:0*
T0*(
_output_shapes
:         аU
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:         аW
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:         аL
Tanh_1Tanh	add_1:z:0*
T0*(
_output_shapes
:         аZ
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*(
_output_shapes
:         аn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :┼
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Й
whileStatelessWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hinit_cstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelrecurrent_kernelbias*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*f
_output_shapesT
R: : : : :         а:         а: : :	ђ	:
ађ	:ђ	* 
_read_only_resource_inputs
 *
_stateful_parallelism( *
bodyR
while_body_1210350*
condR
while_cond_1210349*e
output_shapesT
R: : : : :         а:         а: : :	ђ	:
ађ	:ђ	*
parallel_iterations Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       О
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:         а*
element_dtype0*
num_elementsh
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ѕ
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         а*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ќ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:         а[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  ђ?a
IdentityIdentitystrided_slice_2:output:0*
T0*(
_output_shapes
:         а^

Identity_1Identitytranspose_1:y:0*
T0*,
_output_shapes
:         аY

Identity_2Identitywhile:output:4*
T0*(
_output_shapes
:         аY

Identity_3Identitywhile:output:5*
T0*(
_output_shapes
:         аI

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*p
_input_shapes_
]:         :         а:         а:	ђ	:
ађ	:ђ	*=
api_implements+)lstm_eb22870e-927f-49a2-b687-85a1e58290d1*
api_preferred_deviceCPU*
go_backwards( *

time_major( :S O
+
_output_shapes
:         
 
_user_specified_nameinputs:PL
(
_output_shapes
:         а
 
_user_specified_nameinit_h:PL
(
_output_shapes
:         а
 
_user_specified_nameinit_c:GC

_output_shapes
:	ђ	
 
_user_specified_namekernel:RN
 
_output_shapes
:
ађ	
*
_user_specified_namerecurrent_kernel:A=

_output_shapes	
:ђ	

_user_specified_namebias
№)
л
while_body_1209639
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_matmul_kernel_0%
!while_matmul_1_recurrent_kernel_0
while_biasadd_bias_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_matmul_kernel#
while_matmul_1_recurrent_kernel
while_biasadd_biasѕ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       д
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0њ
while/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while_matmul_kernel_0*
T0*(
_output_shapes
:         ђ	Ѓ
while/MatMul_1MatMulwhile_placeholder_2!while_matmul_1_recurrent_kernel_0*
T0*(
_output_shapes
:         ђ	w
	while/addAddV2while/MatMul:product:0while/MatMul_1:product:0*
T0*(
_output_shapes
:         ђ	p
while/BiasAddBiasAddwhile/add:z:0while_biasadd_bias_0*
T0*(
_output_shapes
:         ђ	W
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :╠
while/splitSplitwhile/split/split_dim:output:0while/BiasAdd:output:0*
T0*d
_output_shapesR
P:         а:         а:         а:         а*
	num_splita
while/SigmoidSigmoidwhile/split:output:0*
T0*(
_output_shapes
:         аc
while/Sigmoid_1Sigmoidwhile/split:output:1*
T0*(
_output_shapes
:         аm
	while/mulMulwhile/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:         а[

while/TanhTanhwhile/split:output:2*
T0*(
_output_shapes
:         аh
while/mul_1Mulwhile/Sigmoid:y:0while/Tanh:y:0*
T0*(
_output_shapes
:         аg
while/add_1AddV2while/mul:z:0while/mul_1:z:0*
T0*(
_output_shapes
:         аc
while/Sigmoid_2Sigmoidwhile/split:output:3*
T0*(
_output_shapes
:         аX
while/Tanh_1Tanhwhile/add_1:z:0*
T0*(
_output_shapes
:         аl
while/mul_2Mulwhile/Sigmoid_2:y:0while/Tanh_1:y:0*
T0*(
_output_shapes
:         аr
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : Я
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/mul_2:z:0*
_output_shapes
: *
element_dtype0:жУмO
while/add_2/yConst*
_output_shapes
: *
dtype0*
value	B :`
while/add_2AddV2while_placeholderwhile/add_2/y:output:0*
T0*
_output_shapes
: O
while/add_3/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_3AddV2while_while_loop_counterwhile/add_3/y:output:0*
T0*
_output_shapes
: L
while/IdentityIdentitywhile/add_3:z:0*
T0*
_output_shapes
: ]
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: N
while/Identity_2Identitywhile/add_2:z:0*
T0*
_output_shapes
: y
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: `
while/Identity_4Identitywhile/mul_2:z:0*
T0*(
_output_shapes
:         а`
while/Identity_5Identitywhile/add_1:z:0*
T0*(
_output_shapes
:         а"*
while_biasadd_biaswhile_biasadd_bias_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"D
while_matmul_1_recurrent_kernel!while_matmul_1_recurrent_kernel_0",
while_matmul_kernelwhile_matmul_kernel_0",
while_strided_slicewhile_strided_slice_0"е
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*e
_input_shapesT
R: : : : :         а:         а: : :	ђ	:
ађ	:ђ	: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:         а:.*
(
_output_shapes
:         а:

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	ђ	:&	"
 
_output_shapes
:
ађ	:!


_output_shapes	
:ђ	
Ч
и
)__inference_lstm_13_layer_call_fn_1212129

inputs
unknown:	ђ	
	unknown_0:
ађ	
	unknown_1:	ђ	
identityѕбStatefulPartitionedCallу
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         а*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_lstm_13_layer_call_and_return_conditional_losses_1209487p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         а`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
Ы┬
у
=__inference___backward_gpu_lstm_with_fallback_1212393_1212569
placeholder
placeholder_1
placeholder_2
placeholder_3
placeholder_43
/gradients_expanddims_2_grad_shape_strided_slice)
%gradients_squeeze_grad_shape_cudnnrnn+
'gradients_squeeze_1_grad_shape_cudnnrnn/
+gradients_strided_slice_grad_shape_cudnnrnn!
gradients_zeros_like_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9
5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_15
1gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h,
(gradients_expanddims_1_grad_shape_init_c-
)gradients_concat_1_grad_mod_concat_1_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_permA
=gradients_transpose_7_grad_invertpermutation_transpose_7_permA
=gradients_transpose_8_grad_invertpermutation_transpose_8_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim)
%gradients_concat_grad_mod_concat_axis
identity

identity_1

identity_2

identity_3

identity_4

identity_5ѕ_
gradients/grad_ys_0Identityplaceholder*
T0*(
_output_shapes
:         аe
gradients/grad_ys_1Identityplaceholder_1*
T0*,
_output_shapes
:         аa
gradients/grad_ys_2Identityplaceholder_2*
T0*(
_output_shapes
:         аa
gradients/grad_ys_3Identityplaceholder_3*
T0*(
_output_shapes
:         аO
gradients/grad_ys_4Identityplaceholder_4*
T0*
_output_shapes
: ђ
!gradients/ExpandDims_2_grad/ShapeShape/gradients_expanddims_2_grad_shape_strided_slice*
T0*
_output_shapes
:Ф
#gradients/ExpandDims_2_grad/ReshapeReshapegradients/grad_ys_1:output:0*gradients/ExpandDims_2_grad/Shape:output:0*
T0*(
_output_shapes
:         аq
gradients/Squeeze_grad/ShapeShape%gradients_squeeze_grad_shape_cudnnrnn*
T0*
_output_shapes
:Ц
gradients/Squeeze_grad/ReshapeReshapegradients/grad_ys_2:output:0%gradients/Squeeze_grad/Shape:output:0*
T0*,
_output_shapes
:         аu
gradients/Squeeze_1_grad/ShapeShape'gradients_squeeze_1_grad_shape_cudnnrnn*
T0*
_output_shapes
:Е
 gradients/Squeeze_1_grad/ReshapeReshapegradients/grad_ys_3:output:0'gradients/Squeeze_1_grad/Shape:output:0*
T0*,
_output_shapes
:         ак
gradients/AddNAddNgradients/grad_ys_0:output:0,gradients/ExpandDims_2_grad/Reshape:output:0*
N*
T0*&
_class
loc:@gradients/grad_ys_0*(
_output_shapes
:         а}
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
:є
3gradients/strided_slice_grad/StridedSliceGrad/beginConst*
_output_shapes
:*
dtype0*
valueB:
         {
1gradients/strided_slice_grad/StridedSliceGrad/endConst*
_output_shapes
:*
dtype0*
valueB: 
5gradients/strided_slice_grad/StridedSliceGrad/stridesConst*
_output_shapes
:*
dtype0*
valueB:Б
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0<gradients/strided_slice_grad/StridedSliceGrad/begin:output:0:gradients/strided_slice_grad/StridedSliceGrad/end:output:0>gradients/strided_slice_grad/StridedSliceGrad/strides:output:0gradients/AddN:sum:0*
Index0*
T0*5
_output_shapes#
!:                  а*
shrink_axis_maskc
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
:«
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_11gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1+gradients_strided_slice_grad_shape_cudnnrnn%gradients_squeeze_grad_shape_cudnnrnn'gradients_squeeze_1_grad_shape_cudnnrnn6gradients/strided_slice_grad/StridedSliceGrad:output:0'gradients/Squeeze_grad/Reshape:output:0)gradients/Squeeze_1_grad/Reshape:output:0gradients_zeros_like_cudnnrnn*
T0*l
_output_shapesZ
X:                  :         а:         а:ђ╗ќ
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:┘
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*4
_output_shapes"
 :                  u
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
:к
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*(
_output_shapes
:         аy
!gradients/ExpandDims_1_grad/ShapeShape(gradients_expanddims_1_grad_shape_init_c*
T0*
_output_shapes
:╩
#gradients/ExpandDims_1_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_c_backprop:0*gradients/ExpandDims_1_grad/Shape:output:0*
T0*(
_output_shapes
:         а^
gradients/concat_1_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :џ
gradients/concat_1_grad/modFloorMod)gradients_concat_1_grad_mod_concat_1_axis%gradients/concat_1_grad/Rank:output:0*
T0*
_output_shapes
: h
gradients/concat_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:аj
gradients/concat_1_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:аj
gradients/concat_1_grad/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:аj
gradients/concat_1_grad/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:аk
gradients/concat_1_grad/Shape_4Const*
_output_shapes
:*
dtype0*
valueB:ђѕk
gradients/concat_1_grad/Shape_5Const*
_output_shapes
:*
dtype0*
valueB:ђѕk
gradients/concat_1_grad/Shape_6Const*
_output_shapes
:*
dtype0*
valueB:ђѕk
gradients/concat_1_grad/Shape_7Const*
_output_shapes
:*
dtype0*
valueB:ђѕj
gradients/concat_1_grad/Shape_8Const*
_output_shapes
:*
dtype0*
valueB:аj
gradients/concat_1_grad/Shape_9Const*
_output_shapes
:*
dtype0*
valueB:аk
 gradients/concat_1_grad/Shape_10Const*
_output_shapes
:*
dtype0*
valueB:аk
 gradients/concat_1_grad/Shape_11Const*
_output_shapes
:*
dtype0*
valueB:аk
 gradients/concat_1_grad/Shape_12Const*
_output_shapes
:*
dtype0*
valueB:аk
 gradients/concat_1_grad/Shape_13Const*
_output_shapes
:*
dtype0*
valueB:аk
 gradients/concat_1_grad/Shape_14Const*
_output_shapes
:*
dtype0*
valueB:аk
 gradients/concat_1_grad/Shape_15Const*
_output_shapes
:*
dtype0*
valueB:аЭ
$gradients/concat_1_grad/ConcatOffsetConcatOffsetgradients/concat_1_grad/mod:z:0&gradients/concat_1_grad/Shape:output:0(gradients/concat_1_grad/Shape_1:output:0(gradients/concat_1_grad/Shape_2:output:0(gradients/concat_1_grad/Shape_3:output:0(gradients/concat_1_grad/Shape_4:output:0(gradients/concat_1_grad/Shape_5:output:0(gradients/concat_1_grad/Shape_6:output:0(gradients/concat_1_grad/Shape_7:output:0(gradients/concat_1_grad/Shape_8:output:0(gradients/concat_1_grad/Shape_9:output:0)gradients/concat_1_grad/Shape_10:output:0)gradients/concat_1_grad/Shape_11:output:0)gradients/concat_1_grad/Shape_12:output:0)gradients/concat_1_grad/Shape_13:output:0)gradients/concat_1_grad/Shape_14:output:0)gradients/concat_1_grad/Shape_15:output:0*
N*t
_output_shapesb
`::::::::::::::::В
gradients/concat_1_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:0&gradients/concat_1_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:а­
gradients/concat_1_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:1(gradients/concat_1_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:а­
gradients/concat_1_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:2(gradients/concat_1_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes	
:а­
gradients/concat_1_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:3(gradients/concat_1_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes	
:аы
gradients/concat_1_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:4(gradients/concat_1_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes

:ђѕы
gradients/concat_1_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:5(gradients/concat_1_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes

:ђѕы
gradients/concat_1_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:6(gradients/concat_1_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes

:ђѕы
gradients/concat_1_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:7(gradients/concat_1_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes

:ђѕ­
gradients/concat_1_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:8(gradients/concat_1_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes	
:а­
gradients/concat_1_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:9(gradients/concat_1_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes	
:аз
 gradients/concat_1_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:10)gradients/concat_1_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes	
:аз
 gradients/concat_1_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:11)gradients/concat_1_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes	
:аз
 gradients/concat_1_grad/Slice_12Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:12)gradients/concat_1_grad/Shape_12:output:0*
Index0*
T0*
_output_shapes	
:аз
 gradients/concat_1_grad/Slice_13Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:13)gradients/concat_1_grad/Shape_13:output:0*
Index0*
T0*
_output_shapes	
:аз
 gradients/concat_1_grad/Slice_14Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:14)gradients/concat_1_grad/Shape_14:output:0*
Index0*
T0*
_output_shapes	
:аз
 gradients/concat_1_grad/Slice_15Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:15)gradients/concat_1_grad/Shape_15:output:0*
Index0*
T0*
_output_shapes	
:аm
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      б
gradients/Reshape_grad/ReshapeReshape&gradients/concat_1_grad/Slice:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes
:	аo
gradients/Reshape_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      е
 gradients/Reshape_1_grad/ReshapeReshape(gradients/concat_1_grad/Slice_1:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0*
_output_shapes
:	аo
gradients/Reshape_2_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      е
 gradients/Reshape_2_grad/ReshapeReshape(gradients/concat_1_grad/Slice_2:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0*
_output_shapes
:	аo
gradients/Reshape_3_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      е
 gradients/Reshape_3_grad/ReshapeReshape(gradients/concat_1_grad/Slice_3:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0*
_output_shapes
:	аo
gradients/Reshape_4_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      Е
 gradients/Reshape_4_grad/ReshapeReshape(gradients/concat_1_grad/Slice_4:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0* 
_output_shapes
:
ааo
gradients/Reshape_5_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      Е
 gradients/Reshape_5_grad/ReshapeReshape(gradients/concat_1_grad/Slice_5:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0* 
_output_shapes
:
ааo
gradients/Reshape_6_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      Е
 gradients/Reshape_6_grad/ReshapeReshape(gradients/concat_1_grad/Slice_6:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0* 
_output_shapes
:
ааo
gradients/Reshape_7_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      Е
 gradients/Reshape_7_grad/ReshapeReshape(gradients/concat_1_grad/Slice_7:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0* 
_output_shapes
:
ааi
gradients/Reshape_8_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:ац
 gradients/Reshape_8_grad/ReshapeReshape(gradients/concat_1_grad/Slice_8:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes	
:аi
gradients/Reshape_9_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:ац
 gradients/Reshape_9_grad/ReshapeReshape(gradients/concat_1_grad/Slice_9:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes	
:аj
gradients/Reshape_10_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:аД
!gradients/Reshape_10_grad/ReshapeReshape)gradients/concat_1_grad/Slice_10:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes	
:аj
gradients/Reshape_11_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:аД
!gradients/Reshape_11_grad/ReshapeReshape)gradients/concat_1_grad/Slice_11:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes	
:аj
gradients/Reshape_12_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:аД
!gradients/Reshape_12_grad/ReshapeReshape)gradients/concat_1_grad/Slice_12:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes	
:аj
gradients/Reshape_13_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:аД
!gradients/Reshape_13_grad/ReshapeReshape)gradients/concat_1_grad/Slice_13:output:0(gradients/Reshape_13_grad/Shape:output:0*
T0*
_output_shapes	
:аj
gradients/Reshape_14_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:аД
!gradients/Reshape_14_grad/ReshapeReshape)gradients/concat_1_grad/Slice_14:output:0(gradients/Reshape_14_grad/Shape:output:0*
T0*
_output_shapes	
:аj
gradients/Reshape_15_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:аД
!gradients/Reshape_15_grad/ReshapeReshape)gradients/concat_1_grad/Slice_15:output:0(gradients/Reshape_15_grad/Shape:output:0*
T0*
_output_shapes	
:аю
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:Х
$gradients/transpose_1_grad/transpose	Transpose'gradients/Reshape_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0*
_output_shapes
:	аю
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:И
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0*
_output_shapes
:	аю
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:И
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0*
_output_shapes
:	аю
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:И
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0*
_output_shapes
:	аю
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:╣
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
ааю
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:╣
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
ааю
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:╣
$gradients/transpose_7_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
ааю
,gradients/transpose_8_grad/InvertPermutationInvertPermutation=gradients_transpose_8_grad_invertpermutation_transpose_8_perm*
_output_shapes
:╣
$gradients/transpose_8_grad/transpose	Transpose)gradients/Reshape_7_grad/Reshape:output:00gradients/transpose_8_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
аау
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0*gradients/Reshape_13_grad/Reshape:output:0*gradients/Reshape_14_grad/Reshape:output:0*gradients/Reshape_15_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
N*
T0*
_output_shapes	
:ђ»
gradients/split_grad/concatConcatV2(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
N*
T0*
_output_shapes
:	ђ	Х
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0(gradients/transpose_7_grad/transpose:y:0(gradients/transpose_8_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
N*
T0* 
_output_shapes
:
ађ	\
gradients/concat_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :њ
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: f
gradients/concat_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:ђ	h
gradients/concat_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:ђ	╩
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0*
N* 
_output_shapes
::м
gradients/concat_grad/SliceSlice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:ђ	о
gradients/concat_grad/Slice_1Slice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:ђ	{
IdentityIdentity&gradients/transpose_grad/transpose:y:0*
T0*4
_output_shapes"
 :                  u

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0*
T0*(
_output_shapes
:         аw

Identity_2Identity,gradients/ExpandDims_1_grad/Reshape:output:0*
T0*(
_output_shapes
:         аf

Identity_3Identity$gradients/split_grad/concat:output:0*
T0*
_output_shapes
:	ђ	i

Identity_4Identity&gradients/split_1_grad/concat:output:0*
T0* 
_output_shapes
:
ађ	d

Identity_5Identity&gradients/concat_grad/Slice_1:output:0*
T0*
_output_shapes	
:ђ	"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*(
_construction_contextkEagerRuntime*ћ
_input_shapesѓ
 :         а:         а:         а:         а: :         а:         а:         а:                  а::                  :         а:         а:ђ╗::         а:         а: ::::::::: : : : *=
api_implements+)lstm_8e4ab49d-552b-4e19-8094-3b0af108a7d4*
api_preferred_deviceGPU*C
forward_function_name*(__forward_gpu_lstm_with_fallback_1212568*
go_backwards( *

time_major( :. *
(
_output_shapes
:         а:2.
,
_output_shapes
:         а:.*
(
_output_shapes
:         а:.*
(
_output_shapes
:         а:

_output_shapes
: :.*
(
_output_shapes
:         а:2.
,
_output_shapes
:         а:2.
,
_output_shapes
:         а:;7
5
_output_shapes#
!:                  а:	

_output_shapes
:::
6
4
_output_shapes"
 :                  :2.
,
_output_shapes
:         а:2.
,
_output_shapes
:         а:"

_output_shapes

:ђ╗: 

_output_shapes
::.*
(
_output_shapes
:         а:.*
(
_output_shapes
:         а:

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
┴B
═
*__inference_gpu_lstm_with_fallback_1213685

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4ѕc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:         P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : p

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0*,
_output_shapes
:         аR
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : t
ExpandDims_1
ExpandDimsinit_cExpandDims_1/dim:output:0*
T0*,
_output_shapes
:         аQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ї
splitSplitsplit/split_dim:output:0kernel*
T0*@
_output_shapes.
,:	а:	а:	а:	а*
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ъ
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*D
_output_shapes2
0:
аа:
аа:
аа:
аа*
	num_splite
zeros_like/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB:ђ	U
zeros_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    x

zeros_likeFill#zeros_like/shape_as_tensor:output:0zeros_like/Const:output:0*
T0*
_output_shapes	
:ђ	M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : r
concatConcatV2zeros_like:output:0biasconcat/axis:output:0*
N*
T0*
_output_shapes	
:ђS
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : Ц
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*L
_output_shapes:
8:а:а:а:а:а:а:а:а*
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
         a
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       m
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes
:	аY
ReshapeReshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:аa
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       m
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes
:	а[
	Reshape_1Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:аa
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       m
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes
:	а[
	Reshape_2Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:аa
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       m
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0*
_output_shapes
:	а[
	Reshape_3Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes	
:аa
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       p
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0* 
_output_shapes
:
аа\
	Reshape_4Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes

:ђѕa
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       p
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0* 
_output_shapes
:
аа\
	Reshape_5Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes

:ђѕa
transpose_7/permConst*
_output_shapes
:*
dtype0*
valueB"       p
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0* 
_output_shapes
:
аа\
	Reshape_6Reshapetranspose_7:y:0Const:output:0*
T0*
_output_shapes

:ђѕa
transpose_8/permConst*
_output_shapes
:*
dtype0*
valueB"       p
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0* 
_output_shapes
:
аа\
	Reshape_7Reshapetranspose_8:y:0Const:output:0*
T0*
_output_shapes

:ђѕ\
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes	
:а\
	Reshape_9Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes	
:а]

Reshape_10Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes	
:а]

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes	
:а]

Reshape_12Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes	
:а]

Reshape_13Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes	
:а]

Reshape_14Reshapesplit_2:output:6Const:output:0*
T0*
_output_shapes	
:а]

Reshape_15Reshapesplit_2:output:7Const:output:0*
T0*
_output_shapes	
:аO
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : а
concat_1ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T0*
_output_shapes

:ђ╗═
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1:output:0*
T0*`
_output_shapesN
L:         а:         а:         а:f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
         _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Т
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         а*
shrink_axis_maskq
SqueezeSqueezeCudnnRNN:output_h:0*
T0*(
_output_shapes
:         а*
squeeze_dims
 s
	Squeeze_1SqueezeCudnnRNN:output_c:0*
T0*(
_output_shapes
:         а*
squeeze_dims
 R
ExpandDims_2/dimConst*
_output_shapes
: *
dtype0*
value	B :ё
ExpandDims_2
ExpandDimsstrided_slice:output:0ExpandDims_2/dim:output:0*
T0*,
_output_shapes
:         а[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @_
IdentityIdentitystrided_slice:output:0*
T0*(
_output_shapes
:         аd

Identity_1IdentityExpandDims_2:output:0*
T0*,
_output_shapes
:         а[

Identity_2IdentitySqueeze:output:0*
T0*(
_output_shapes
:         а]

Identity_3IdentitySqueeze_1:output:0*
T0*(
_output_shapes
:         аI

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*p
_input_shapes_
]:         :         а:         а:	ђ	:
ађ	:ђ	*=
api_implements+)lstm_1013b0ea-424a-45a4-86e4-0f580df52117*
api_preferred_deviceGPU*
go_backwards( *

time_major( :S O
+
_output_shapes
:         
 
_user_specified_nameinputs:PL
(
_output_shapes
:         а
 
_user_specified_nameinit_h:PL
(
_output_shapes
:         а
 
_user_specified_nameinit_c:GC

_output_shapes
:	ђ	
 
_user_specified_namekernel:RN
 
_output_shapes
:
ађ	
*
_user_specified_namerecurrent_kernel:A=

_output_shapes	
:ђ	

_user_specified_namebias
Е
я
L__inference_change_layer_12_layer_call_and_return_conditional_losses_1210231
input_1(
sequential_13_1210218:	ђ	)
sequential_13_1210220:
ађ	$
sequential_13_1210222:	ђ	(
sequential_13_1210224:	а#
sequential_13_1210226:
identityѕб%sequential_13/StatefulPartitionedCallО
%sequential_13/StatefulPartitionedCallStatefulPartitionedCallinput_1sequential_13_1210218sequential_13_1210220sequential_13_1210222sequential_13_1210224sequential_13_1210226*
Tin

2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         *'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8ѓ *S
fNRL
J__inference_sequential_13_layer_call_and_return_conditional_losses_1210043{
addAddV2input_1.sequential_13/StatefulPartitionedCall:output:0*
T0*+
_output_shapes
:         Z
IdentityIdentityadd:z:0^NoOp*
T0*+
_output_shapes
:         n
NoOpNoOp&^sequential_13/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:         : : : : : 2N
%sequential_13/StatefulPartitionedCall%sequential_13/StatefulPartitionedCall:T P
+
_output_shapes
:         
!
_user_specified_name	input_1
Б
┐
D__inference_lstm_13_layer_call_and_return_conditional_losses_1213433

inputs/
read_readvariableop_resource:	ђ	2
read_1_readvariableop_resource:
ађ	-
read_2_readvariableop_resource:	ђ	

identity_3ѕбRead/ReadVariableOpбRead_1/ReadVariableOpбRead_2/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Л
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :аs
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:         аS
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :аw
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    s
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:         аq
Read/ReadVariableOpReadVariableOpread_readvariableop_resource*
_output_shapes
:	ђ	*
dtype0[
IdentityIdentityRead/ReadVariableOp:value:0*
T0*
_output_shapes
:	ђ	v
Read_1/ReadVariableOpReadVariableOpread_1_readvariableop_resource* 
_output_shapes
:
ађ	*
dtype0`

Identity_1IdentityRead_1/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ађ	q
Read_2/ReadVariableOpReadVariableOpread_2_readvariableop_resource*
_output_shapes	
:ђ	*
dtype0[

Identity_2IdentityRead_2/ReadVariableOp:value:0*
T0*
_output_shapes	
:ђ	и
PartitionedCallPartitionedCallinputszeros:output:0zeros_1:output:0Identity:output:0Identity_1:output:0Identity_2:output:0*
Tin

2*
Tout	
2*
_collective_manager_ids
 *j
_output_shapesX
V:         а:         а:         а:         а: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ **
f%R#
!__inference_standard_lstm_1213158j

Identity_3IdentityPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         аї
NoOpNoOp^Read/ReadVariableOp^Read_1/ReadVariableOp^Read_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         : : : 2*
Read/ReadVariableOpRead/ReadVariableOp2.
Read_1/ReadVariableOpRead_1/ReadVariableOp2.
Read_2/ReadVariableOpRead_2/ReadVariableOp:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
Ф
H
,__inference_reshape_13_layer_call_fn_1213888

inputs
identityХ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *P
fKRI
G__inference_reshape_13_layer_call_and_return_conditional_losses_1209524d
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
ш.
ф
J__inference_sequential_13_layer_call_and_return_conditional_losses_1211650

inputs7
$lstm_13_read_readvariableop_resource:	ђ	:
&lstm_13_read_1_readvariableop_resource:
ађ	5
&lstm_13_read_2_readvariableop_resource:	ђ	:
'dense_13_matmul_readvariableop_resource:	а6
(dense_13_biasadd_readvariableop_resource:
identityѕбdense_13/BiasAdd/ReadVariableOpбdense_13/MatMul/ReadVariableOpбlstm_13/Read/ReadVariableOpбlstm_13/Read_1/ReadVariableOpбlstm_13/Read_2/ReadVariableOpC
lstm_13/ShapeShapeinputs*
T0*
_output_shapes
:e
lstm_13/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: g
lstm_13/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
lstm_13/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:щ
lstm_13/strided_sliceStridedSlicelstm_13/Shape:output:0$lstm_13/strided_slice/stack:output:0&lstm_13/strided_slice/stack_1:output:0&lstm_13/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskY
lstm_13/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :аІ
lstm_13/zeros/packedPacklstm_13/strided_slice:output:0lstm_13/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:X
lstm_13/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Ё
lstm_13/zerosFilllstm_13/zeros/packed:output:0lstm_13/zeros/Const:output:0*
T0*(
_output_shapes
:         а[
lstm_13/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :аЈ
lstm_13/zeros_1/packedPacklstm_13/strided_slice:output:0!lstm_13/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:Z
lstm_13/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    І
lstm_13/zeros_1Filllstm_13/zeros_1/packed:output:0lstm_13/zeros_1/Const:output:0*
T0*(
_output_shapes
:         аЂ
lstm_13/Read/ReadVariableOpReadVariableOp$lstm_13_read_readvariableop_resource*
_output_shapes
:	ђ	*
dtype0k
lstm_13/IdentityIdentity#lstm_13/Read/ReadVariableOp:value:0*
T0*
_output_shapes
:	ђ	є
lstm_13/Read_1/ReadVariableOpReadVariableOp&lstm_13_read_1_readvariableop_resource* 
_output_shapes
:
ађ	*
dtype0p
lstm_13/Identity_1Identity%lstm_13/Read_1/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ађ	Ђ
lstm_13/Read_2/ReadVariableOpReadVariableOp&lstm_13_read_2_readvariableop_resource*
_output_shapes	
:ђ	*
dtype0k
lstm_13/Identity_2Identity%lstm_13/Read_2/ReadVariableOp:value:0*
T0*
_output_shapes	
:ђ	у
lstm_13/PartitionedCallPartitionedCallinputslstm_13/zeros:output:0lstm_13/zeros_1:output:0lstm_13/Identity:output:0lstm_13/Identity_1:output:0lstm_13/Identity_2:output:0*
Tin

2*
Tout	
2*
_collective_manager_ids
 *j
_output_shapesX
V:         а:         а:         а:         а: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ **
f%R#
!__inference_standard_lstm_1211360Є
dense_13/MatMul/ReadVariableOpReadVariableOp'dense_13_matmul_readvariableop_resource*
_output_shapes
:	а*
dtype0Ћ
dense_13/MatMulMatMul lstm_13/PartitionedCall:output:0&dense_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         ё
dense_13/BiasAdd/ReadVariableOpReadVariableOp(dense_13_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Љ
dense_13/BiasAddBiasAdddense_13/MatMul:product:0'dense_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         Y
reshape_13/ShapeShapedense_13/BiasAdd:output:0*
T0*
_output_shapes
:h
reshape_13/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: j
 reshape_13/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:j
 reshape_13/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ѕ
reshape_13/strided_sliceStridedSlicereshape_13/Shape:output:0'reshape_13/strided_slice/stack:output:0)reshape_13/strided_slice/stack_1:output:0)reshape_13/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask\
reshape_13/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :\
reshape_13/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :╗
reshape_13/Reshape/shapePack!reshape_13/strided_slice:output:0#reshape_13/Reshape/shape/1:output:0#reshape_13/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:Љ
reshape_13/ReshapeReshapedense_13/BiasAdd:output:0!reshape_13/Reshape/shape:output:0*
T0*+
_output_shapes
:         n
IdentityIdentityreshape_13/Reshape:output:0^NoOp*
T0*+
_output_shapes
:         у
NoOpNoOp ^dense_13/BiasAdd/ReadVariableOp^dense_13/MatMul/ReadVariableOp^lstm_13/Read/ReadVariableOp^lstm_13/Read_1/ReadVariableOp^lstm_13/Read_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:         : : : : : 2B
dense_13/BiasAdd/ReadVariableOpdense_13/BiasAdd/ReadVariableOp2@
dense_13/MatMul/ReadVariableOpdense_13/MatMul/ReadVariableOp2:
lstm_13/Read/ReadVariableOplstm_13/Read/ReadVariableOp2>
lstm_13/Read_1/ReadVariableOplstm_13/Read_1/ReadVariableOp2>
lstm_13/Read_2/ReadVariableOplstm_13/Read_2/ReadVariableOp:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
┴B
═
*__inference_gpu_lstm_with_fallback_1210532

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4ѕc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:         P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : p

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0*,
_output_shapes
:         аR
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : t
ExpandDims_1
ExpandDimsinit_cExpandDims_1/dim:output:0*
T0*,
_output_shapes
:         аQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ї
splitSplitsplit/split_dim:output:0kernel*
T0*@
_output_shapes.
,:	а:	а:	а:	а*
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ъ
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*D
_output_shapes2
0:
аа:
аа:
аа:
аа*
	num_splite
zeros_like/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB:ђ	U
zeros_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    x

zeros_likeFill#zeros_like/shape_as_tensor:output:0zeros_like/Const:output:0*
T0*
_output_shapes	
:ђ	M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : r
concatConcatV2zeros_like:output:0biasconcat/axis:output:0*
N*
T0*
_output_shapes	
:ђS
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : Ц
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*L
_output_shapes:
8:а:а:а:а:а:а:а:а*
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
         a
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       m
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes
:	аY
ReshapeReshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:аa
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       m
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes
:	а[
	Reshape_1Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:аa
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       m
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes
:	а[
	Reshape_2Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:аa
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       m
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0*
_output_shapes
:	а[
	Reshape_3Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes	
:аa
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       p
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0* 
_output_shapes
:
аа\
	Reshape_4Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes

:ђѕa
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       p
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0* 
_output_shapes
:
аа\
	Reshape_5Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes

:ђѕa
transpose_7/permConst*
_output_shapes
:*
dtype0*
valueB"       p
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0* 
_output_shapes
:
аа\
	Reshape_6Reshapetranspose_7:y:0Const:output:0*
T0*
_output_shapes

:ђѕa
transpose_8/permConst*
_output_shapes
:*
dtype0*
valueB"       p
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0* 
_output_shapes
:
аа\
	Reshape_7Reshapetranspose_8:y:0Const:output:0*
T0*
_output_shapes

:ђѕ\
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes	
:а\
	Reshape_9Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes	
:а]

Reshape_10Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes	
:а]

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes	
:а]

Reshape_12Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes	
:а]

Reshape_13Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes	
:а]

Reshape_14Reshapesplit_2:output:6Const:output:0*
T0*
_output_shapes	
:а]

Reshape_15Reshapesplit_2:output:7Const:output:0*
T0*
_output_shapes	
:аO
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : а
concat_1ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T0*
_output_shapes

:ђ╗═
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1:output:0*
T0*`
_output_shapesN
L:         а:         а:         а:f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
         _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Т
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         а*
shrink_axis_maskq
SqueezeSqueezeCudnnRNN:output_h:0*
T0*(
_output_shapes
:         а*
squeeze_dims
 s
	Squeeze_1SqueezeCudnnRNN:output_c:0*
T0*(
_output_shapes
:         а*
squeeze_dims
 R
ExpandDims_2/dimConst*
_output_shapes
: *
dtype0*
value	B :ё
ExpandDims_2
ExpandDimsstrided_slice:output:0ExpandDims_2/dim:output:0*
T0*,
_output_shapes
:         а[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @_
IdentityIdentitystrided_slice:output:0*
T0*(
_output_shapes
:         аd

Identity_1IdentityExpandDims_2:output:0*
T0*,
_output_shapes
:         а[

Identity_2IdentitySqueeze:output:0*
T0*(
_output_shapes
:         а]

Identity_3IdentitySqueeze_1:output:0*
T0*(
_output_shapes
:         аI

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*p
_input_shapes_
]:         :         а:         а:	ђ	:
ађ	:ђ	*=
api_implements+)lstm_eb22870e-927f-49a2-b687-85a1e58290d1*
api_preferred_deviceGPU*
go_backwards( *

time_major( :S O
+
_output_shapes
:         
 
_user_specified_nameinputs:PL
(
_output_shapes
:         а
 
_user_specified_nameinit_h:PL
(
_output_shapes
:         а
 
_user_specified_nameinit_c:GC

_output_shapes
:	ђ	
 
_user_specified_namekernel:RN
 
_output_shapes
:
ађ	
*
_user_specified_namerecurrent_kernel:A=

_output_shapes	
:ђ	

_user_specified_namebias
ш.
ф
J__inference_sequential_13_layer_call_and_return_conditional_losses_1212096

inputs7
$lstm_13_read_readvariableop_resource:	ђ	:
&lstm_13_read_1_readvariableop_resource:
ађ	5
&lstm_13_read_2_readvariableop_resource:	ђ	:
'dense_13_matmul_readvariableop_resource:	а6
(dense_13_biasadd_readvariableop_resource:
identityѕбdense_13/BiasAdd/ReadVariableOpбdense_13/MatMul/ReadVariableOpбlstm_13/Read/ReadVariableOpбlstm_13/Read_1/ReadVariableOpбlstm_13/Read_2/ReadVariableOpC
lstm_13/ShapeShapeinputs*
T0*
_output_shapes
:e
lstm_13/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: g
lstm_13/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
lstm_13/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:щ
lstm_13/strided_sliceStridedSlicelstm_13/Shape:output:0$lstm_13/strided_slice/stack:output:0&lstm_13/strided_slice/stack_1:output:0&lstm_13/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskY
lstm_13/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :аІ
lstm_13/zeros/packedPacklstm_13/strided_slice:output:0lstm_13/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:X
lstm_13/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Ё
lstm_13/zerosFilllstm_13/zeros/packed:output:0lstm_13/zeros/Const:output:0*
T0*(
_output_shapes
:         а[
lstm_13/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :аЈ
lstm_13/zeros_1/packedPacklstm_13/strided_slice:output:0!lstm_13/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:Z
lstm_13/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    І
lstm_13/zeros_1Filllstm_13/zeros_1/packed:output:0lstm_13/zeros_1/Const:output:0*
T0*(
_output_shapes
:         аЂ
lstm_13/Read/ReadVariableOpReadVariableOp$lstm_13_read_readvariableop_resource*
_output_shapes
:	ђ	*
dtype0k
lstm_13/IdentityIdentity#lstm_13/Read/ReadVariableOp:value:0*
T0*
_output_shapes
:	ђ	є
lstm_13/Read_1/ReadVariableOpReadVariableOp&lstm_13_read_1_readvariableop_resource* 
_output_shapes
:
ађ	*
dtype0p
lstm_13/Identity_1Identity%lstm_13/Read_1/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ађ	Ђ
lstm_13/Read_2/ReadVariableOpReadVariableOp&lstm_13_read_2_readvariableop_resource*
_output_shapes	
:ђ	*
dtype0k
lstm_13/Identity_2Identity%lstm_13/Read_2/ReadVariableOp:value:0*
T0*
_output_shapes	
:ђ	у
lstm_13/PartitionedCallPartitionedCallinputslstm_13/zeros:output:0lstm_13/zeros_1:output:0lstm_13/Identity:output:0lstm_13/Identity_1:output:0lstm_13/Identity_2:output:0*
Tin

2*
Tout	
2*
_collective_manager_ids
 *j
_output_shapesX
V:         а:         а:         а:         а: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ **
f%R#
!__inference_standard_lstm_1211806Є
dense_13/MatMul/ReadVariableOpReadVariableOp'dense_13_matmul_readvariableop_resource*
_output_shapes
:	а*
dtype0Ћ
dense_13/MatMulMatMul lstm_13/PartitionedCall:output:0&dense_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         ё
dense_13/BiasAdd/ReadVariableOpReadVariableOp(dense_13_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Љ
dense_13/BiasAddBiasAdddense_13/MatMul:product:0'dense_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         Y
reshape_13/ShapeShapedense_13/BiasAdd:output:0*
T0*
_output_shapes
:h
reshape_13/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: j
 reshape_13/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:j
 reshape_13/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ѕ
reshape_13/strided_sliceStridedSlicereshape_13/Shape:output:0'reshape_13/strided_slice/stack:output:0)reshape_13/strided_slice/stack_1:output:0)reshape_13/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask\
reshape_13/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :\
reshape_13/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :╗
reshape_13/Reshape/shapePack!reshape_13/strided_slice:output:0#reshape_13/Reshape/shape/1:output:0#reshape_13/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:Љ
reshape_13/ReshapeReshapedense_13/BiasAdd:output:0!reshape_13/Reshape/shape:output:0*
T0*+
_output_shapes
:         n
IdentityIdentityreshape_13/Reshape:output:0^NoOp*
T0*+
_output_shapes
:         у
NoOpNoOp ^dense_13/BiasAdd/ReadVariableOp^dense_13/MatMul/ReadVariableOp^lstm_13/Read/ReadVariableOp^lstm_13/Read_1/ReadVariableOp^lstm_13/Read_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:         : : : : : 2B
dense_13/BiasAdd/ReadVariableOpdense_13/BiasAdd/ReadVariableOp2@
dense_13/MatMul/ReadVariableOpdense_13/MatMul/ReadVariableOp2:
lstm_13/Read/ReadVariableOplstm_13/Read/ReadVariableOp2>
lstm_13/Read_1/ReadVariableOplstm_13/Read_1/ReadVariableOp2>
lstm_13/Read_2/ReadVariableOplstm_13/Read_2/ReadVariableOp:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
Ф:
И
L__inference_change_layer_12_layer_call_and_return_conditional_losses_1211174

inputsE
2sequential_13_lstm_13_read_readvariableop_resource:	ђ	H
4sequential_13_lstm_13_read_1_readvariableop_resource:
ађ	C
4sequential_13_lstm_13_read_2_readvariableop_resource:	ђ	H
5sequential_13_dense_13_matmul_readvariableop_resource:	аD
6sequential_13_dense_13_biasadd_readvariableop_resource:
identityѕб-sequential_13/dense_13/BiasAdd/ReadVariableOpб,sequential_13/dense_13/MatMul/ReadVariableOpб)sequential_13/lstm_13/Read/ReadVariableOpб+sequential_13/lstm_13/Read_1/ReadVariableOpб+sequential_13/lstm_13/Read_2/ReadVariableOpQ
sequential_13/lstm_13/ShapeShapeinputs*
T0*
_output_shapes
:s
)sequential_13/lstm_13/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+sequential_13/lstm_13/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+sequential_13/lstm_13/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:┐
#sequential_13/lstm_13/strided_sliceStridedSlice$sequential_13/lstm_13/Shape:output:02sequential_13/lstm_13/strided_slice/stack:output:04sequential_13/lstm_13/strided_slice/stack_1:output:04sequential_13/lstm_13/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskg
$sequential_13/lstm_13/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :ах
"sequential_13/lstm_13/zeros/packedPack,sequential_13/lstm_13/strided_slice:output:0-sequential_13/lstm_13/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:f
!sequential_13/lstm_13/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    »
sequential_13/lstm_13/zerosFill+sequential_13/lstm_13/zeros/packed:output:0*sequential_13/lstm_13/zeros/Const:output:0*
T0*(
_output_shapes
:         аi
&sequential_13/lstm_13/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :а╣
$sequential_13/lstm_13/zeros_1/packedPack,sequential_13/lstm_13/strided_slice:output:0/sequential_13/lstm_13/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:h
#sequential_13/lstm_13/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    х
sequential_13/lstm_13/zeros_1Fill-sequential_13/lstm_13/zeros_1/packed:output:0,sequential_13/lstm_13/zeros_1/Const:output:0*
T0*(
_output_shapes
:         аЮ
)sequential_13/lstm_13/Read/ReadVariableOpReadVariableOp2sequential_13_lstm_13_read_readvariableop_resource*
_output_shapes
:	ђ	*
dtype0Є
sequential_13/lstm_13/IdentityIdentity1sequential_13/lstm_13/Read/ReadVariableOp:value:0*
T0*
_output_shapes
:	ђ	б
+sequential_13/lstm_13/Read_1/ReadVariableOpReadVariableOp4sequential_13_lstm_13_read_1_readvariableop_resource* 
_output_shapes
:
ађ	*
dtype0ї
 sequential_13/lstm_13/Identity_1Identity3sequential_13/lstm_13/Read_1/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ађ	Ю
+sequential_13/lstm_13/Read_2/ReadVariableOpReadVariableOp4sequential_13_lstm_13_read_2_readvariableop_resource*
_output_shapes	
:ђ	*
dtype0Є
 sequential_13/lstm_13/Identity_2Identity3sequential_13/lstm_13/Read_2/ReadVariableOp:value:0*
T0*
_output_shapes	
:ђ	╗
%sequential_13/lstm_13/PartitionedCallPartitionedCallinputs$sequential_13/lstm_13/zeros:output:0&sequential_13/lstm_13/zeros_1:output:0'sequential_13/lstm_13/Identity:output:0)sequential_13/lstm_13/Identity_1:output:0)sequential_13/lstm_13/Identity_2:output:0*
Tin

2*
Tout	
2*
_collective_manager_ids
 *j
_output_shapesX
V:         а:         а:         а:         а: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ **
f%R#
!__inference_standard_lstm_1210883Б
,sequential_13/dense_13/MatMul/ReadVariableOpReadVariableOp5sequential_13_dense_13_matmul_readvariableop_resource*
_output_shapes
:	а*
dtype0┐
sequential_13/dense_13/MatMulMatMul.sequential_13/lstm_13/PartitionedCall:output:04sequential_13/dense_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         а
-sequential_13/dense_13/BiasAdd/ReadVariableOpReadVariableOp6sequential_13_dense_13_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0╗
sequential_13/dense_13/BiasAddBiasAdd'sequential_13/dense_13/MatMul:product:05sequential_13/dense_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         u
sequential_13/reshape_13/ShapeShape'sequential_13/dense_13/BiasAdd:output:0*
T0*
_output_shapes
:v
,sequential_13/reshape_13/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: x
.sequential_13/reshape_13/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.sequential_13/reshape_13/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╬
&sequential_13/reshape_13/strided_sliceStridedSlice'sequential_13/reshape_13/Shape:output:05sequential_13/reshape_13/strided_slice/stack:output:07sequential_13/reshape_13/strided_slice/stack_1:output:07sequential_13/reshape_13/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskj
(sequential_13/reshape_13/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :j
(sequential_13/reshape_13/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :з
&sequential_13/reshape_13/Reshape/shapePack/sequential_13/reshape_13/strided_slice:output:01sequential_13/reshape_13/Reshape/shape/1:output:01sequential_13/reshape_13/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:╗
 sequential_13/reshape_13/ReshapeReshape'sequential_13/dense_13/BiasAdd:output:0/sequential_13/reshape_13/Reshape/shape:output:0*
T0*+
_output_shapes
:         u
addAddV2inputs)sequential_13/reshape_13/Reshape:output:0*
T0*+
_output_shapes
:         Z
IdentityIdentityadd:z:0^NoOp*
T0*+
_output_shapes
:         Г
NoOpNoOp.^sequential_13/dense_13/BiasAdd/ReadVariableOp-^sequential_13/dense_13/MatMul/ReadVariableOp*^sequential_13/lstm_13/Read/ReadVariableOp,^sequential_13/lstm_13/Read_1/ReadVariableOp,^sequential_13/lstm_13/Read_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:         : : : : : 2^
-sequential_13/dense_13/BiasAdd/ReadVariableOp-sequential_13/dense_13/BiasAdd/ReadVariableOp2\
,sequential_13/dense_13/MatMul/ReadVariableOp,sequential_13/dense_13/MatMul/ReadVariableOp2V
)sequential_13/lstm_13/Read/ReadVariableOp)sequential_13/lstm_13/Read/ReadVariableOp2Z
+sequential_13/lstm_13/Read_1/ReadVariableOp+sequential_13/lstm_13/Read_1/ReadVariableOp2Z
+sequential_13/lstm_13/Read_2/ReadVariableOp+sequential_13/lstm_13/Read_2/ReadVariableOp:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
њN
б
(__forward_gpu_lstm_with_fallback_1212568

inputs
init_h_0
init_c_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4
strided_slice
cudnnrnn

cudnnrnn_0

cudnnrnn_1

cudnnrnn_2
	transpose

expanddims
expanddims_1
concat_1
transpose_perm

init_h

init_c
concat_1_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
transpose_7_perm
transpose_8_perm
split_2_split_dim
split_split_dim
split_1_split_dim
concat_axisѕc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          B
transpose_0	Transposeinputstranspose/perm:output:0*
T0P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : r

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T0*,
_output_shapes
:         аR
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : v
ExpandDims_1
ExpandDimsinit_c_0ExpandDims_1/dim:output:0*
T0*,
_output_shapes
:         аQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ї
splitSplitsplit/split_dim:output:0kernel*
T0*@
_output_shapes.
,:	а:	а:	а:	а*
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ъ
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*D
_output_shapes2
0:
аа:
аа:
аа:
аа*
	num_splite
zeros_like/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB:ђ	U
zeros_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    x

zeros_likeFill#zeros_like/shape_as_tensor:output:0zeros_like/Const:output:0*
T0*
_output_shapes	
:ђ	M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : r
concatConcatV2zeros_like:output:0biasconcat/axis:output:0*
N*
T0*
_output_shapes	
:ђS
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : Ц
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*L
_output_shapes:
8:а:а:а:а:а:а:а:а*
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
         a
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       m
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes
:	аY
ReshapeReshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:аa
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       m
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes
:	а[
	Reshape_1Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:аa
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       m
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes
:	а[
	Reshape_2Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:аa
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       m
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0*
_output_shapes
:	а[
	Reshape_3Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes	
:аa
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       p
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0* 
_output_shapes
:
аа\
	Reshape_4Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes

:ђѕa
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       p
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0* 
_output_shapes
:
аа\
	Reshape_5Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes

:ђѕa
transpose_7/permConst*
_output_shapes
:*
dtype0*
valueB"       p
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0* 
_output_shapes
:
аа\
	Reshape_6Reshapetranspose_7:y:0Const:output:0*
T0*
_output_shapes

:ђѕa
transpose_8/permConst*
_output_shapes
:*
dtype0*
valueB"       p
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0* 
_output_shapes
:
аа\
	Reshape_7Reshapetranspose_8:y:0Const:output:0*
T0*
_output_shapes

:ђѕ\
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes	
:а\
	Reshape_9Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes	
:а]

Reshape_10Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes	
:а]

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes	
:а]

Reshape_12Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes	
:а]

Reshape_13Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes	
:а]

Reshape_14Reshapesplit_2:output:6Const:output:0*
T0*
_output_shapes	
:а]

Reshape_15Reshapesplit_2:output:7Const:output:0*
T0*
_output_shapes	
:аO
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ё

concat_1_0ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T0┌
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1_0:output:0*
T0*i
_output_shapesW
U:                  а:         а:         а:f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
         _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Й
strided_slice_0StridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
shrink_axis_maskq
SqueezeSqueezeCudnnRNN:output_h:0*
T0*(
_output_shapes
:         а*
squeeze_dims
 s
	Squeeze_1SqueezeCudnnRNN:output_c:0*
T0*(
_output_shapes
:         а*
squeeze_dims
 R
ExpandDims_2/dimConst*
_output_shapes
: *
dtype0*
value	B :є
ExpandDims_2
ExpandDimsstrided_slice_0:output:0ExpandDims_2/dim:output:0*
T0*,
_output_shapes
:         а[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @a
IdentityIdentitystrided_slice_0:output:0*
T0*(
_output_shapes
:         аd

Identity_1IdentityExpandDims_2:output:0*
T0*,
_output_shapes
:         а[

Identity_2IdentitySqueeze:output:0*
T0*(
_output_shapes
:         а]

Identity_3IdentitySqueeze_1:output:0*
T0*(
_output_shapes
:         аI

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: "
concat_1concat_1_0:output:0"'
concat_1_axisconcat_1/axis:output:0"#
concat_axisconcat/axis:output:0"
cudnnrnnCudnnRNN:output_h:0"!

cudnnrnn_0CudnnRNN:output_c:0"

cudnnrnn_1CudnnRNN:output:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"!

expanddimsExpandDims:output:0"%
expanddims_1ExpandDims_1:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
init_cinit_c_0"
init_hinit_h_0"/
split_1_split_dimsplit_1/split_dim:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"+
split_split_dimsplit/split_dim:output:0")
strided_slicestrided_slice_0:output:0"
	transposetranspose_0:y:0"-
transpose_1_permtranspose_1/perm:output:0"-
transpose_2_permtranspose_2/perm:output:0"-
transpose_3_permtranspose_3/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"-
transpose_5_permtranspose_5/perm:output:0"-
transpose_6_permtranspose_6/perm:output:0"-
transpose_7_permtranspose_7/perm:output:0"-
transpose_8_permtranspose_8/perm:output:0")
transpose_permtranspose/perm:output:0*(
_construction_contextkEagerRuntime*y
_input_shapesh
f:                  :         а:         а:	ђ	:
ађ	:ђ	*=
api_implements+)lstm_8e4ab49d-552b-4e19-8094-3b0af108a7d4*
api_preferred_deviceGPU*Y
backward_function_name?=__inference___backward_gpu_lstm_with_fallback_1212393_1212569*
go_backwards( *

time_major( :\ X
4
_output_shapes"
 :                  
 
_user_specified_nameinputs:PL
(
_output_shapes
:         а
 
_user_specified_nameinit_h:PL
(
_output_shapes
:         а
 
_user_specified_nameinit_c:GC

_output_shapes
:	ђ	
 
_user_specified_namekernel:RN
 
_output_shapes
:
ађ	
*
_user_specified_namerecurrent_kernel:A=

_output_shapes	
:ђ	

_user_specified_namebias
тB
═
*__inference_gpu_lstm_with_fallback_1212823

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4ѕc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :                  P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : p

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0*,
_output_shapes
:         аR
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : t
ExpandDims_1
ExpandDimsinit_cExpandDims_1/dim:output:0*
T0*,
_output_shapes
:         аQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ї
splitSplitsplit/split_dim:output:0kernel*
T0*@
_output_shapes.
,:	а:	а:	а:	а*
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ъ
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*D
_output_shapes2
0:
аа:
аа:
аа:
аа*
	num_splite
zeros_like/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB:ђ	U
zeros_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    x

zeros_likeFill#zeros_like/shape_as_tensor:output:0zeros_like/Const:output:0*
T0*
_output_shapes	
:ђ	M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : r
concatConcatV2zeros_like:output:0biasconcat/axis:output:0*
N*
T0*
_output_shapes	
:ђS
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : Ц
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*L
_output_shapes:
8:а:а:а:а:а:а:а:а*
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
         a
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       m
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes
:	аY
ReshapeReshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:аa
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       m
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes
:	а[
	Reshape_1Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:аa
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       m
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes
:	а[
	Reshape_2Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:аa
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       m
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0*
_output_shapes
:	а[
	Reshape_3Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes	
:аa
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       p
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0* 
_output_shapes
:
аа\
	Reshape_4Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes

:ђѕa
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       p
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0* 
_output_shapes
:
аа\
	Reshape_5Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes

:ђѕa
transpose_7/permConst*
_output_shapes
:*
dtype0*
valueB"       p
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0* 
_output_shapes
:
аа\
	Reshape_6Reshapetranspose_7:y:0Const:output:0*
T0*
_output_shapes

:ђѕa
transpose_8/permConst*
_output_shapes
:*
dtype0*
valueB"       p
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0* 
_output_shapes
:
аа\
	Reshape_7Reshapetranspose_8:y:0Const:output:0*
T0*
_output_shapes

:ђѕ\
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes	
:а\
	Reshape_9Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes	
:а]

Reshape_10Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes	
:а]

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes	
:а]

Reshape_12Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes	
:а]

Reshape_13Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes	
:а]

Reshape_14Reshapesplit_2:output:6Const:output:0*
T0*
_output_shapes	
:а]

Reshape_15Reshapesplit_2:output:7Const:output:0*
T0*
_output_shapes	
:аO
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : а
concat_1ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T0*
_output_shapes

:ђ╗о
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1:output:0*
T0*i
_output_shapesW
U:                  а:         а:         а:f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
         _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Т
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         а*
shrink_axis_maskq
SqueezeSqueezeCudnnRNN:output_h:0*
T0*(
_output_shapes
:         а*
squeeze_dims
 s
	Squeeze_1SqueezeCudnnRNN:output_c:0*
T0*(
_output_shapes
:         а*
squeeze_dims
 R
ExpandDims_2/dimConst*
_output_shapes
: *
dtype0*
value	B :ё
ExpandDims_2
ExpandDimsstrided_slice:output:0ExpandDims_2/dim:output:0*
T0*,
_output_shapes
:         а[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @_
IdentityIdentitystrided_slice:output:0*
T0*(
_output_shapes
:         аd

Identity_1IdentityExpandDims_2:output:0*
T0*,
_output_shapes
:         а[

Identity_2IdentitySqueeze:output:0*
T0*(
_output_shapes
:         а]

Identity_3IdentitySqueeze_1:output:0*
T0*(
_output_shapes
:         аI

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*y
_input_shapesh
f:                  :         а:         а:	ђ	:
ађ	:ђ	*=
api_implements+)lstm_bc8f3e75-dc57-4569-96f0-8e80ed216f21*
api_preferred_deviceGPU*
go_backwards( *

time_major( :\ X
4
_output_shapes"
 :                  
 
_user_specified_nameinputs:PL
(
_output_shapes
:         а
 
_user_specified_nameinit_h:PL
(
_output_shapes
:         а
 
_user_specified_nameinit_c:GC

_output_shapes
:	ђ	
 
_user_specified_namekernel:RN
 
_output_shapes
:
ађ	
*
_user_specified_namerecurrent_kernel:A=

_output_shapes	
:ђ	

_user_specified_namebias"є
L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*│
serving_defaultЪ
?
input_14
serving_default_input_1:0         @
output_14
StatefulPartitionedCall:0         tensorflow/serving/predict:│╔
В
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	model
		optimizer


signatures"
_tf_keras_model
C
0
1
2
3
4"
trackable_list_wrapper
C
0
1
2
3
4"
trackable_list_wrapper
 "
trackable_list_wrapper
╩
non_trainable_variables

layers
metrics
layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
Ѓ
trace_0
trace_1
trace_2
trace_32ў
1__inference_change_layer_12_layer_call_fn_1210138
1__inference_change_layer_12_layer_call_fn_1210265
1__inference_change_layer_12_layer_call_fn_1210280
1__inference_change_layer_12_layer_call_fn_1210199╔
└▓╝
FullArgSpec
argsџ
jself
jinputs
varargsjargs
varkwjkwargs
defaults
 

kwonlyargsџ

jtraining%
kwonlydefaultsф

trainingp 
annotationsф *
 ztrace_0ztrace_1ztrace_2ztrace_3
№
trace_0
trace_1
trace_2
trace_32ё
L__inference_change_layer_12_layer_call_and_return_conditional_losses_1210727
L__inference_change_layer_12_layer_call_and_return_conditional_losses_1211174
L__inference_change_layer_12_layer_call_and_return_conditional_losses_1210215
L__inference_change_layer_12_layer_call_and_return_conditional_losses_1210231╔
└▓╝
FullArgSpec
argsџ
jself
jinputs
varargsjargs
varkwjkwargs
defaults
 

kwonlyargsџ

jtraining%
kwonlydefaultsф

trainingp 
annotationsф *
 ztrace_0ztrace_1ztrace_2ztrace_3
═B╩
"__inference__wrapped_model_1208164input_1"ў
Љ▓Ї
FullArgSpec
argsџ 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Ё
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
 	variables
!trainable_variables
"regularization_losses
#	keras_api
$__call__
*%&call_and_return_all_conditional_losses"
_tf_keras_sequential
ю
&
_variables
'_iterations
(_learning_rate
)_index_dict
*
_momentums
+_velocities
,_update_step_xla"
experimentalOptimizer
,
-serving_default"
signature_map
.:,	ђ	2lstm_13/lstm_cell_13/kernel
9:7
ађ	2%lstm_13/lstm_cell_13/recurrent_kernel
(:&ђ	2lstm_13/lstm_cell_13/bias
": 	а2dense_13/kernel
:2dense_13/bias
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
5
.0
/1
02"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ЇBі
1__inference_change_layer_12_layer_call_fn_1210138input_1"╔
└▓╝
FullArgSpec
argsџ
jself
jinputs
varargsjargs
varkwjkwargs
defaults
 

kwonlyargsџ

jtraining%
kwonlydefaultsф

trainingp 
annotationsф *
 
їBЅ
1__inference_change_layer_12_layer_call_fn_1210265inputs"╔
└▓╝
FullArgSpec
argsџ
jself
jinputs
varargsjargs
varkwjkwargs
defaults
 

kwonlyargsџ

jtraining%
kwonlydefaultsф

trainingp 
annotationsф *
 
їBЅ
1__inference_change_layer_12_layer_call_fn_1210280inputs"╔
└▓╝
FullArgSpec
argsџ
jself
jinputs
varargsjargs
varkwjkwargs
defaults
 

kwonlyargsџ

jtraining%
kwonlydefaultsф

trainingp 
annotationsф *
 
ЇBі
1__inference_change_layer_12_layer_call_fn_1210199input_1"╔
└▓╝
FullArgSpec
argsџ
jself
jinputs
varargsjargs
varkwjkwargs
defaults
 

kwonlyargsџ

jtraining%
kwonlydefaultsф

trainingp 
annotationsф *
 
ДBц
L__inference_change_layer_12_layer_call_and_return_conditional_losses_1210727inputs"╔
└▓╝
FullArgSpec
argsџ
jself
jinputs
varargsjargs
varkwjkwargs
defaults
 

kwonlyargsџ

jtraining%
kwonlydefaultsф

trainingp 
annotationsф *
 
ДBц
L__inference_change_layer_12_layer_call_and_return_conditional_losses_1211174inputs"╔
└▓╝
FullArgSpec
argsџ
jself
jinputs
varargsjargs
varkwjkwargs
defaults
 

kwonlyargsџ

jtraining%
kwonlydefaultsф

trainingp 
annotationsф *
 
еBЦ
L__inference_change_layer_12_layer_call_and_return_conditional_losses_1210215input_1"╔
└▓╝
FullArgSpec
argsџ
jself
jinputs
varargsjargs
varkwjkwargs
defaults
 

kwonlyargsџ

jtraining%
kwonlydefaultsф

trainingp 
annotationsф *
 
еBЦ
L__inference_change_layer_12_layer_call_and_return_conditional_losses_1210231input_1"╔
└▓╝
FullArgSpec
argsџ
jself
jinputs
varargsjargs
varkwjkwargs
defaults
 

kwonlyargsџ

jtraining%
kwonlydefaultsф

trainingp 
annotationsф *
 
┌
1	variables
2trainable_variables
3regularization_losses
4	keras_api
5__call__
*6&call_and_return_all_conditional_losses
7_random_generator
8cell
9
state_spec"
_tf_keras_rnn_layer
╗
:	variables
;trainable_variables
<regularization_losses
=	keras_api
>__call__
*?&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
Ц
@	variables
Atrainable_variables
Bregularization_losses
C	keras_api
D__call__
*E&call_and_return_all_conditional_losses"
_tf_keras_layer
C
0
1
2
3
4"
trackable_list_wrapper
C
0
1
2
3
4"
trackable_list_wrapper
 "
trackable_list_wrapper
Г
Fnon_trainable_variables

Glayers
Hmetrics
Ilayer_regularization_losses
Jlayer_metrics
 	variables
!trainable_variables
"regularization_losses
$__call__
*%&call_and_return_all_conditional_losses
&%"call_and_return_conditional_losses"
_generic_user_object
ы
Ktrace_0
Ltrace_1
Mtrace_2
Ntrace_32є
/__inference_sequential_13_layer_call_fn_1209540
/__inference_sequential_13_layer_call_fn_1211189
/__inference_sequential_13_layer_call_fn_1211204
/__inference_sequential_13_layer_call_fn_1210071┐
Х▓▓
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zKtrace_0zLtrace_1zMtrace_2zNtrace_3
П
Otrace_0
Ptrace_1
Qtrace_2
Rtrace_32Ы
J__inference_sequential_13_layer_call_and_return_conditional_losses_1211650
J__inference_sequential_13_layer_call_and_return_conditional_losses_1212096
J__inference_sequential_13_layer_call_and_return_conditional_losses_1210088
J__inference_sequential_13_layer_call_and_return_conditional_losses_1210105┐
Х▓▓
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zOtrace_0zPtrace_1zQtrace_2zRtrace_3
n
'0
S1
T2
U3
V4
W5
X6
Y7
Z8
[9
\10"
trackable_list_wrapper
:	 2	iteration
: 2learning_rate
 "
trackable_dict_wrapper
C
S0
U1
W2
Y3
[4"
trackable_list_wrapper
C
T0
V1
X2
Z3
\4"
trackable_list_wrapper
┐2╝╣
«▓ф
FullArgSpec2
args*џ'
jself

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 0
╠B╔
%__inference_signature_wrapper_1210250input_1"ћ
Ї▓Ѕ
FullArgSpec
argsџ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
N
]	variables
^	keras_api
	_total
	`count"
_tf_keras_metric
N
a	variables
b	keras_api
	ctotal
	dcount"
_tf_keras_metric
^
e	variables
f	keras_api
	gtotal
	hcount
i
_fn_kwargs"
_tf_keras_metric
5
0
1
2"
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
╣

jstates
knon_trainable_variables

llayers
mmetrics
nlayer_regularization_losses
olayer_metrics
1	variables
2trainable_variables
3regularization_losses
5__call__
*6&call_and_return_all_conditional_losses
&6"call_and_return_conditional_losses"
_generic_user_object
Ь
ptrace_0
qtrace_1
rtrace_2
strace_32Ѓ
)__inference_lstm_13_layer_call_fn_1212107
)__inference_lstm_13_layer_call_fn_1212118
)__inference_lstm_13_layer_call_fn_1212129
)__inference_lstm_13_layer_call_fn_1212140н
╦▓К
FullArgSpecB
args:џ7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsџ

 
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zptrace_0zqtrace_1zrtrace_2zstrace_3
┌
ttrace_0
utrace_1
vtrace_2
wtrace_32№
D__inference_lstm_13_layer_call_and_return_conditional_losses_1212571
D__inference_lstm_13_layer_call_and_return_conditional_losses_1213002
D__inference_lstm_13_layer_call_and_return_conditional_losses_1213433
D__inference_lstm_13_layer_call_and_return_conditional_losses_1213864н
╦▓К
FullArgSpecB
args:џ7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsџ

 
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zttrace_0zutrace_1zvtrace_2zwtrace_3
"
_generic_user_object
Э
x	variables
ytrainable_variables
zregularization_losses
{	keras_api
|__call__
*}&call_and_return_all_conditional_losses
~_random_generator

state_size

kernel
recurrent_kernel
bias"
_tf_keras_layer
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
ђnon_trainable_variables
Ђlayers
ѓmetrics
 Ѓlayer_regularization_losses
ёlayer_metrics
:	variables
;trainable_variables
<regularization_losses
>__call__
*?&call_and_return_all_conditional_losses
&?"call_and_return_conditional_losses"
_generic_user_object
­
Ёtrace_02Л
*__inference_dense_13_layer_call_fn_1213873б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zЁtrace_0
І
єtrace_02В
E__inference_dense_13_layer_call_and_return_conditional_losses_1213883б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zєtrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▓
Єnon_trainable_variables
ѕlayers
Ѕmetrics
 іlayer_regularization_losses
Іlayer_metrics
@	variables
Atrainable_variables
Bregularization_losses
D__call__
*E&call_and_return_all_conditional_losses
&E"call_and_return_conditional_losses"
_generic_user_object
Ы
їtrace_02М
,__inference_reshape_13_layer_call_fn_1213888б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zїtrace_0
Ї
Їtrace_02Ь
G__inference_reshape_13_layer_call_and_return_conditional_losses_1213901б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zЇtrace_0
 "
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ЄBё
/__inference_sequential_13_layer_call_fn_1209540lstm_13_input"┐
Х▓▓
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ђB§
/__inference_sequential_13_layer_call_fn_1211189inputs"┐
Х▓▓
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ђB§
/__inference_sequential_13_layer_call_fn_1211204inputs"┐
Х▓▓
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЄBё
/__inference_sequential_13_layer_call_fn_1210071lstm_13_input"┐
Х▓▓
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЏBў
J__inference_sequential_13_layer_call_and_return_conditional_losses_1211650inputs"┐
Х▓▓
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЏBў
J__inference_sequential_13_layer_call_and_return_conditional_losses_1212096inputs"┐
Х▓▓
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
бBЪ
J__inference_sequential_13_layer_call_and_return_conditional_losses_1210088lstm_13_input"┐
Х▓▓
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
бBЪ
J__inference_sequential_13_layer_call_and_return_conditional_losses_1210105lstm_13_input"┐
Х▓▓
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
3:1	ђ	2"Adam/m/lstm_13/lstm_cell_13/kernel
3:1	ђ	2"Adam/v/lstm_13/lstm_cell_13/kernel
>:<
ађ	2,Adam/m/lstm_13/lstm_cell_13/recurrent_kernel
>:<
ађ	2,Adam/v/lstm_13/lstm_cell_13/recurrent_kernel
-:+ђ	2 Adam/m/lstm_13/lstm_cell_13/bias
-:+ђ	2 Adam/v/lstm_13/lstm_cell_13/bias
':%	а2Adam/m/dense_13/kernel
':%	а2Adam/v/dense_13/kernel
 :2Adam/m/dense_13/bias
 :2Adam/v/dense_13/bias
.
_0
`1"
trackable_list_wrapper
-
]	variables"
_generic_user_object
:  (2total
:  (2count
.
c0
d1"
trackable_list_wrapper
-
a	variables"
_generic_user_object
:  (2total
:  (2count
.
g0
h1"
trackable_list_wrapper
-
e	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
80"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ЉBј
)__inference_lstm_13_layer_call_fn_1212107inputs_0"н
╦▓К
FullArgSpecB
args:џ7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsџ

 
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЉBј
)__inference_lstm_13_layer_call_fn_1212118inputs_0"н
╦▓К
FullArgSpecB
args:џ7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsџ

 
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЈBї
)__inference_lstm_13_layer_call_fn_1212129inputs"н
╦▓К
FullArgSpecB
args:џ7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsџ

 
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЈBї
)__inference_lstm_13_layer_call_fn_1212140inputs"н
╦▓К
FullArgSpecB
args:џ7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsџ

 
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
гBЕ
D__inference_lstm_13_layer_call_and_return_conditional_losses_1212571inputs_0"н
╦▓К
FullArgSpecB
args:џ7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsџ

 
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
гBЕ
D__inference_lstm_13_layer_call_and_return_conditional_losses_1213002inputs_0"н
╦▓К
FullArgSpecB
args:џ7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsџ

 
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
фBД
D__inference_lstm_13_layer_call_and_return_conditional_losses_1213433inputs"н
╦▓К
FullArgSpecB
args:џ7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsџ

 
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
фBД
D__inference_lstm_13_layer_call_and_return_conditional_losses_1213864inputs"н
╦▓К
FullArgSpecB
args:џ7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsџ

 
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
5
0
1
2"
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
јnon_trainable_variables
Јlayers
љmetrics
 Љlayer_regularization_losses
њlayer_metrics
x	variables
ytrainable_variables
zregularization_losses
|__call__
*}&call_and_return_all_conditional_losses
&}"call_and_return_conditional_losses"
_generic_user_object
├2└й
┤▓░
FullArgSpec3
args+џ(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
├2└й
┤▓░
FullArgSpec3
args+џ(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
"
_generic_user_object
 "
trackable_list_wrapper
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
яB█
*__inference_dense_13_layer_call_fn_1213873inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
щBШ
E__inference_dense_13_layer_call_and_return_conditional_losses_1213883inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
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
ЯBП
,__inference_reshape_13_layer_call_fn_1213888inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
чBЭ
G__inference_reshape_13_layer_call_and_return_conditional_losses_1213901inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
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
trackable_dict_wrapperю
"__inference__wrapped_model_1208164v4б1
*б'
%і"
input_1         
ф "7ф4
2
output_1&і#
output_1         ¤
L__inference_change_layer_12_layer_call_and_return_conditional_losses_1210215DбA
*б'
%і"
input_1         
ф

trainingp "0б-
&і#
tensor_0         
џ ¤
L__inference_change_layer_12_layer_call_and_return_conditional_losses_1210231DбA
*б'
%і"
input_1         
ф

trainingp"0б-
&і#
tensor_0         
џ ╬
L__inference_change_layer_12_layer_call_and_return_conditional_losses_1210727~Cб@
)б&
$і!
inputs         
ф

trainingp "0б-
&і#
tensor_0         
џ ╬
L__inference_change_layer_12_layer_call_and_return_conditional_losses_1211174~Cб@
)б&
$і!
inputs         
ф

trainingp"0б-
&і#
tensor_0         
џ Е
1__inference_change_layer_12_layer_call_fn_1210138tDбA
*б'
%і"
input_1         
ф

trainingp "%і"
unknown         Е
1__inference_change_layer_12_layer_call_fn_1210199tDбA
*б'
%і"
input_1         
ф

trainingp"%і"
unknown         е
1__inference_change_layer_12_layer_call_fn_1210265sCб@
)б&
$і!
inputs         
ф

trainingp "%і"
unknown         е
1__inference_change_layer_12_layer_call_fn_1210280sCб@
)б&
$і!
inputs         
ф

trainingp"%і"
unknown         Г
E__inference_dense_13_layer_call_and_return_conditional_losses_1213883d0б-
&б#
!і
inputs         а
ф ",б)
"і
tensor_0         
џ Є
*__inference_dense_13_layer_call_fn_1213873Y0б-
&б#
!і
inputs         а
ф "!і
unknown         ╬
D__inference_lstm_13_layer_call_and_return_conditional_losses_1212571ЁOбL
EбB
4џ1
/і,
inputs_0                  

 
p 

 
ф "-б*
#і 
tensor_0         а
џ ╬
D__inference_lstm_13_layer_call_and_return_conditional_losses_1213002ЁOбL
EбB
4џ1
/і,
inputs_0                  

 
p

 
ф "-б*
#і 
tensor_0         а
џ й
D__inference_lstm_13_layer_call_and_return_conditional_losses_1213433u?б<
5б2
$і!
inputs         

 
p 

 
ф "-б*
#і 
tensor_0         а
џ й
D__inference_lstm_13_layer_call_and_return_conditional_losses_1213864u?б<
5б2
$і!
inputs         

 
p

 
ф "-б*
#і 
tensor_0         а
џ Д
)__inference_lstm_13_layer_call_fn_1212107zOбL
EбB
4џ1
/і,
inputs_0                  

 
p 

 
ф ""і
unknown         аД
)__inference_lstm_13_layer_call_fn_1212118zOбL
EбB
4џ1
/і,
inputs_0                  

 
p

 
ф ""і
unknown         аЌ
)__inference_lstm_13_layer_call_fn_1212129j?б<
5б2
$і!
inputs         

 
p 

 
ф ""і
unknown         аЌ
)__inference_lstm_13_layer_call_fn_1212140j?б<
5б2
$і!
inputs         

 
p

 
ф ""і
unknown         а«
G__inference_reshape_13_layer_call_and_return_conditional_losses_1213901c/б,
%б"
 і
inputs         
ф "0б-
&і#
tensor_0         
џ ѕ
,__inference_reshape_13_layer_call_fn_1213888X/б,
%б"
 і
inputs         
ф "%і"
unknown         ╦
J__inference_sequential_13_layer_call_and_return_conditional_losses_1210088}Bб?
8б5
+і(
lstm_13_input         
p 

 
ф "0б-
&і#
tensor_0         
џ ╦
J__inference_sequential_13_layer_call_and_return_conditional_losses_1210105}Bб?
8б5
+і(
lstm_13_input         
p

 
ф "0б-
&і#
tensor_0         
џ ─
J__inference_sequential_13_layer_call_and_return_conditional_losses_1211650v;б8
1б.
$і!
inputs         
p 

 
ф "0б-
&і#
tensor_0         
џ ─
J__inference_sequential_13_layer_call_and_return_conditional_losses_1212096v;б8
1б.
$і!
inputs         
p

 
ф "0б-
&і#
tensor_0         
џ Ц
/__inference_sequential_13_layer_call_fn_1209540rBб?
8б5
+і(
lstm_13_input         
p 

 
ф "%і"
unknown         Ц
/__inference_sequential_13_layer_call_fn_1210071rBб?
8б5
+і(
lstm_13_input         
p

 
ф "%і"
unknown         ъ
/__inference_sequential_13_layer_call_fn_1211189k;б8
1б.
$і!
inputs         
p 

 
ф "%і"
unknown         ъ
/__inference_sequential_13_layer_call_fn_1211204k;б8
1б.
$і!
inputs         
p

 
ф "%і"
unknown         Ф
%__inference_signature_wrapper_1210250Ђ?б<
б 
5ф2
0
input_1%і"
input_1         "7ф4
2
output_1&і#
output_1         