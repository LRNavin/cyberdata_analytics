       �K"	   �B=�Abrain.Event:2����     ��$	Va?�B=�A"��
j
input_1Placeholder*
dtype0*'
_output_shapes
:���������+*
shape:���������+
m
dense_1/random_uniform/shapeConst*
valueB"+   
   *
dtype0*
_output_shapes
:
_
dense_1/random_uniform/minConst*
valueB
 *�D��*
dtype0*
_output_shapes
: 
_
dense_1/random_uniform/maxConst*
valueB
 *�D�>*
dtype0*
_output_shapes
: 
�
$dense_1/random_uniform/RandomUniformRandomUniformdense_1/random_uniform/shape*
dtype0*
_output_shapes

:+
*
seed2끞*

seed*
T0
z
dense_1/random_uniform/subSubdense_1/random_uniform/maxdense_1/random_uniform/min*
T0*
_output_shapes
: 
�
dense_1/random_uniform/mulMul$dense_1/random_uniform/RandomUniformdense_1/random_uniform/sub*
T0*
_output_shapes

:+

~
dense_1/random_uniformAdddense_1/random_uniform/muldense_1/random_uniform/min*
T0*
_output_shapes

:+

�
dense_1/kernel
VariableV2*
shared_name *
dtype0*
_output_shapes

:+
*
	container *
shape
:+

�
dense_1/kernel/AssignAssigndense_1/kerneldense_1/random_uniform*
use_locking(*
T0*!
_class
loc:@dense_1/kernel*
validate_shape(*
_output_shapes

:+

{
dense_1/kernel/readIdentitydense_1/kernel*
T0*!
_class
loc:@dense_1/kernel*
_output_shapes

:+

Z
dense_1/ConstConst*
dtype0*
_output_shapes
:
*
valueB
*    
x
dense_1/bias
VariableV2*
shape:
*
shared_name *
dtype0*
_output_shapes
:
*
	container 
�
dense_1/bias/AssignAssigndense_1/biasdense_1/Const*
use_locking(*
T0*
_class
loc:@dense_1/bias*
validate_shape(*
_output_shapes
:

q
dense_1/bias/readIdentitydense_1/bias*
_output_shapes
:
*
T0*
_class
loc:@dense_1/bias
�
dense_1/MatMulMatMulinput_1dense_1/kernel/read*'
_output_shapes
:���������
*
transpose_a( *
transpose_b( *
T0
�
dense_1/BiasAddBiasAdddense_1/MatMuldense_1/bias/read*'
_output_shapes
:���������
*
T0*
data_formatNHWC
W
dense_1/TanhTanhdense_1/BiasAdd*
T0*'
_output_shapes
:���������

g
 dense_1/activity_regularizer/AbsAbsdense_1/Tanh*
T0*'
_output_shapes
:���������

g
"dense_1/activity_regularizer/mul/xConst*
valueB
 *o�:*
dtype0*
_output_shapes
: 
�
 dense_1/activity_regularizer/mulMul"dense_1/activity_regularizer/mul/x dense_1/activity_regularizer/Abs*
T0*'
_output_shapes
:���������

s
"dense_1/activity_regularizer/ConstConst*
dtype0*
_output_shapes
:*
valueB"       
�
 dense_1/activity_regularizer/SumSum dense_1/activity_regularizer/mul"dense_1/activity_regularizer/Const*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
g
"dense_1/activity_regularizer/add/xConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
 dense_1/activity_regularizer/addAdd"dense_1/activity_regularizer/add/x dense_1/activity_regularizer/Sum*
T0*
_output_shapes
: 
m
dense_2/random_uniform/shapeConst*
valueB"
      *
dtype0*
_output_shapes
:
_
dense_2/random_uniform/minConst*
valueB
 *��!�*
dtype0*
_output_shapes
: 
_
dense_2/random_uniform/maxConst*
valueB
 *��!?*
dtype0*
_output_shapes
: 
�
$dense_2/random_uniform/RandomUniformRandomUniformdense_2/random_uniform/shape*
T0*
dtype0*
_output_shapes

:
*
seed2�ߜ*

seed
z
dense_2/random_uniform/subSubdense_2/random_uniform/maxdense_2/random_uniform/min*
_output_shapes
: *
T0
�
dense_2/random_uniform/mulMul$dense_2/random_uniform/RandomUniformdense_2/random_uniform/sub*
_output_shapes

:
*
T0
~
dense_2/random_uniformAdddense_2/random_uniform/muldense_2/random_uniform/min*
T0*
_output_shapes

:

�
dense_2/kernel
VariableV2*
shared_name *
dtype0*
_output_shapes

:
*
	container *
shape
:

�
dense_2/kernel/AssignAssigndense_2/kerneldense_2/random_uniform*
use_locking(*
T0*!
_class
loc:@dense_2/kernel*
validate_shape(*
_output_shapes

:

{
dense_2/kernel/readIdentitydense_2/kernel*
T0*!
_class
loc:@dense_2/kernel*
_output_shapes

:

Z
dense_2/ConstConst*
valueB*    *
dtype0*
_output_shapes
:
x
dense_2/bias
VariableV2*
dtype0*
_output_shapes
:*
	container *
shape:*
shared_name 
�
dense_2/bias/AssignAssigndense_2/biasdense_2/Const*
use_locking(*
T0*
_class
loc:@dense_2/bias*
validate_shape(*
_output_shapes
:
q
dense_2/bias/readIdentitydense_2/bias*
_class
loc:@dense_2/bias*
_output_shapes
:*
T0
�
dense_2/MatMulMatMuldense_1/Tanhdense_2/kernel/read*'
_output_shapes
:���������*
transpose_a( *
transpose_b( *
T0
�
dense_2/BiasAddBiasAdddense_2/MatMuldense_2/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:���������
W
dense_2/ReluReludense_2/BiasAdd*'
_output_shapes
:���������*
T0
m
dense_3/random_uniform/shapeConst*
dtype0*
_output_shapes
:*
valueB"      
_
dense_3/random_uniform/minConst*
valueB
 *�KF�*
dtype0*
_output_shapes
: 
_
dense_3/random_uniform/maxConst*
valueB
 *�KF?*
dtype0*
_output_shapes
: 
�
$dense_3/random_uniform/RandomUniformRandomUniformdense_3/random_uniform/shape*

seed*
T0*
dtype0*
_output_shapes

:*
seed2��
z
dense_3/random_uniform/subSubdense_3/random_uniform/maxdense_3/random_uniform/min*
_output_shapes
: *
T0
�
dense_3/random_uniform/mulMul$dense_3/random_uniform/RandomUniformdense_3/random_uniform/sub*
_output_shapes

:*
T0
~
dense_3/random_uniformAdddense_3/random_uniform/muldense_3/random_uniform/min*
T0*
_output_shapes

:
�
dense_3/kernel
VariableV2*
dtype0*
_output_shapes

:*
	container *
shape
:*
shared_name 
�
dense_3/kernel/AssignAssigndense_3/kerneldense_3/random_uniform*
use_locking(*
T0*!
_class
loc:@dense_3/kernel*
validate_shape(*
_output_shapes

:
{
dense_3/kernel/readIdentitydense_3/kernel*!
_class
loc:@dense_3/kernel*
_output_shapes

:*
T0
Z
dense_3/ConstConst*
valueB*    *
dtype0*
_output_shapes
:
x
dense_3/bias
VariableV2*
shape:*
shared_name *
dtype0*
_output_shapes
:*
	container 
�
dense_3/bias/AssignAssigndense_3/biasdense_3/Const*
T0*
_class
loc:@dense_3/bias*
validate_shape(*
_output_shapes
:*
use_locking(
q
dense_3/bias/readIdentitydense_3/bias*
T0*
_class
loc:@dense_3/bias*
_output_shapes
:
�
dense_3/MatMulMatMuldense_2/Reludense_3/kernel/read*'
_output_shapes
:���������*
transpose_a( *
transpose_b( *
T0
�
dense_3/BiasAddBiasAdddense_3/MatMuldense_3/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:���������
W
dense_3/TanhTanhdense_3/BiasAdd*'
_output_shapes
:���������*
T0
m
dense_4/random_uniform/shapeConst*
valueB"   +   *
dtype0*
_output_shapes
:
_
dense_4/random_uniform/minConst*
valueB
 *���*
dtype0*
_output_shapes
: 
_
dense_4/random_uniform/maxConst*
valueB
 *��>*
dtype0*
_output_shapes
: 
�
$dense_4/random_uniform/RandomUniformRandomUniformdense_4/random_uniform/shape*
T0*
dtype0*
_output_shapes

:+*
seed2�Ֆ*

seed
z
dense_4/random_uniform/subSubdense_4/random_uniform/maxdense_4/random_uniform/min*
T0*
_output_shapes
: 
�
dense_4/random_uniform/mulMul$dense_4/random_uniform/RandomUniformdense_4/random_uniform/sub*
T0*
_output_shapes

:+
~
dense_4/random_uniformAdddense_4/random_uniform/muldense_4/random_uniform/min*
T0*
_output_shapes

:+
�
dense_4/kernel
VariableV2*
dtype0*
_output_shapes

:+*
	container *
shape
:+*
shared_name 
�
dense_4/kernel/AssignAssigndense_4/kerneldense_4/random_uniform*
use_locking(*
T0*!
_class
loc:@dense_4/kernel*
validate_shape(*
_output_shapes

:+
{
dense_4/kernel/readIdentitydense_4/kernel*
T0*!
_class
loc:@dense_4/kernel*
_output_shapes

:+
Z
dense_4/ConstConst*
dtype0*
_output_shapes
:+*
valueB+*    
x
dense_4/bias
VariableV2*
shared_name *
dtype0*
_output_shapes
:+*
	container *
shape:+
�
dense_4/bias/AssignAssigndense_4/biasdense_4/Const*
use_locking(*
T0*
_class
loc:@dense_4/bias*
validate_shape(*
_output_shapes
:+
q
dense_4/bias/readIdentitydense_4/bias*
T0*
_class
loc:@dense_4/bias*
_output_shapes
:+
�
dense_4/MatMulMatMuldense_3/Tanhdense_4/kernel/read*
T0*'
_output_shapes
:���������+*
transpose_a( *
transpose_b( 
�
dense_4/BiasAddBiasAdddense_4/MatMuldense_4/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:���������+
W
dense_4/ReluReludense_4/BiasAdd*'
_output_shapes
:���������+*
T0
_
Adam/iterations/initial_valueConst*
value	B	 R *
dtype0	*
_output_shapes
: 
s
Adam/iterations
VariableV2*
shape: *
shared_name *
dtype0	*
_output_shapes
: *
	container 
�
Adam/iterations/AssignAssignAdam/iterationsAdam/iterations/initial_value*
use_locking(*
T0	*"
_class
loc:@Adam/iterations*
validate_shape(*
_output_shapes
: 
v
Adam/iterations/readIdentityAdam/iterations*
_output_shapes
: *
T0	*"
_class
loc:@Adam/iterations
Z
Adam/lr/initial_valueConst*
valueB
 *o�:*
dtype0*
_output_shapes
: 
k
Adam/lr
VariableV2*
dtype0*
_output_shapes
: *
	container *
shape: *
shared_name 
�
Adam/lr/AssignAssignAdam/lrAdam/lr/initial_value*
T0*
_class
loc:@Adam/lr*
validate_shape(*
_output_shapes
: *
use_locking(
^
Adam/lr/readIdentityAdam/lr*
T0*
_class
loc:@Adam/lr*
_output_shapes
: 
^
Adam/beta_1/initial_valueConst*
valueB
 *fff?*
dtype0*
_output_shapes
: 
o
Adam/beta_1
VariableV2*
shape: *
shared_name *
dtype0*
_output_shapes
: *
	container 
�
Adam/beta_1/AssignAssignAdam/beta_1Adam/beta_1/initial_value*
T0*
_class
loc:@Adam/beta_1*
validate_shape(*
_output_shapes
: *
use_locking(
j
Adam/beta_1/readIdentityAdam/beta_1*
_output_shapes
: *
T0*
_class
loc:@Adam/beta_1
^
Adam/beta_2/initial_valueConst*
valueB
 *w�?*
dtype0*
_output_shapes
: 
o
Adam/beta_2
VariableV2*
dtype0*
_output_shapes
: *
	container *
shape: *
shared_name 
�
Adam/beta_2/AssignAssignAdam/beta_2Adam/beta_2/initial_value*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@Adam/beta_2
j
Adam/beta_2/readIdentityAdam/beta_2*
_class
loc:@Adam/beta_2*
_output_shapes
: *
T0
]
Adam/decay/initial_valueConst*
_output_shapes
: *
valueB
 *    *
dtype0
n

Adam/decay
VariableV2*
dtype0*
_output_shapes
: *
	container *
shape: *
shared_name 
�
Adam/decay/AssignAssign
Adam/decayAdam/decay/initial_value*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@Adam/decay
g
Adam/decay/readIdentity
Adam/decay*
T0*
_class
loc:@Adam/decay*
_output_shapes
: 
�
dense_4_targetPlaceholder*
dtype0*0
_output_shapes
:������������������*%
shape:������������������
q
dense_4_sample_weightsPlaceholder*
dtype0*#
_output_shapes
:���������*
shape:���������
l
loss/dense_4_loss/subSubdense_4/Reludense_4_target*
T0*'
_output_shapes
:���������+
k
loss/dense_4_loss/SquareSquareloss/dense_4_loss/sub*
T0*'
_output_shapes
:���������+
s
(loss/dense_4_loss/Mean/reduction_indicesConst*
_output_shapes
: *
valueB :
���������*
dtype0
�
loss/dense_4_loss/MeanMeanloss/dense_4_loss/Square(loss/dense_4_loss/Mean/reduction_indices*#
_output_shapes
:���������*
	keep_dims( *

Tidx0*
T0
m
*loss/dense_4_loss/Mean_1/reduction_indicesConst*
valueB *
dtype0*
_output_shapes
: 
�
loss/dense_4_loss/Mean_1Meanloss/dense_4_loss/Mean*loss/dense_4_loss/Mean_1/reduction_indices*
T0*#
_output_shapes
:���������*
	keep_dims( *

Tidx0
|
loss/dense_4_loss/mulMulloss/dense_4_loss/Mean_1dense_4_sample_weights*#
_output_shapes
:���������*
T0
a
loss/dense_4_loss/NotEqual/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
loss/dense_4_loss/NotEqualNotEqualdense_4_sample_weightsloss/dense_4_loss/NotEqual/y*
T0*#
_output_shapes
:���������
�
loss/dense_4_loss/CastCastloss/dense_4_loss/NotEqual*

SrcT0
*
Truncate( *#
_output_shapes
:���������*

DstT0
a
loss/dense_4_loss/ConstConst*
dtype0*
_output_shapes
:*
valueB: 
�
loss/dense_4_loss/Mean_2Meanloss/dense_4_loss/Castloss/dense_4_loss/Const*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
�
loss/dense_4_loss/truedivRealDivloss/dense_4_loss/mulloss/dense_4_loss/Mean_2*
T0*#
_output_shapes
:���������
c
loss/dense_4_loss/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
�
loss/dense_4_loss/Mean_3Meanloss/dense_4_loss/truedivloss/dense_4_loss/Const_1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
O

loss/mul/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
V
loss/mulMul
loss/mul/xloss/dense_4_loss/Mean_3*
T0*
_output_shapes
: 
\
loss/addAddloss/mul dense_1/activity_regularizer/add*
_output_shapes
: *
T0
g
metrics/acc/ArgMax/dimensionConst*
valueB :
���������*
dtype0*
_output_shapes
: 
�
metrics/acc/ArgMaxArgMaxdense_4_targetmetrics/acc/ArgMax/dimension*

Tidx0*
T0*
output_type0	*#
_output_shapes
:���������
i
metrics/acc/ArgMax_1/dimensionConst*
_output_shapes
: *
valueB :
���������*
dtype0
�
metrics/acc/ArgMax_1ArgMaxdense_4/Relumetrics/acc/ArgMax_1/dimension*
output_type0	*#
_output_shapes
:���������*

Tidx0*
T0
r
metrics/acc/EqualEqualmetrics/acc/ArgMaxmetrics/acc/ArgMax_1*#
_output_shapes
:���������*
T0	
x
metrics/acc/CastCastmetrics/acc/Equal*

SrcT0
*
Truncate( *#
_output_shapes
:���������*

DstT0
[
metrics/acc/ConstConst*
valueB: *
dtype0*
_output_shapes
:
{
metrics/acc/MeanMeanmetrics/acc/Castmetrics/acc/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
}
training/Adam/gradients/ShapeConst*
_output_shapes
: *
valueB *
_class
loc:@loss/add*
dtype0
�
!training/Adam/gradients/grad_ys_0Const*
valueB
 *  �?*
_class
loc:@loss/add*
dtype0*
_output_shapes
: 
�
training/Adam/gradients/FillFilltraining/Adam/gradients/Shape!training/Adam/gradients/grad_ys_0*
T0*

index_type0*
_class
loc:@loss/add*
_output_shapes
: 
�
)training/Adam/gradients/loss/mul_grad/MulMultraining/Adam/gradients/Fillloss/dense_4_loss/Mean_3*
T0*
_class
loc:@loss/mul*
_output_shapes
: 
�
+training/Adam/gradients/loss/mul_grad/Mul_1Multraining/Adam/gradients/Fill
loss/mul/x*
_class
loc:@loss/mul*
_output_shapes
: *
T0
�
Ctraining/Adam/gradients/loss/dense_4_loss/Mean_3_grad/Reshape/shapeConst*
valueB:*+
_class!
loc:@loss/dense_4_loss/Mean_3*
dtype0*
_output_shapes
:
�
=training/Adam/gradients/loss/dense_4_loss/Mean_3_grad/ReshapeReshape+training/Adam/gradients/loss/mul_grad/Mul_1Ctraining/Adam/gradients/loss/dense_4_loss/Mean_3_grad/Reshape/shape*
_output_shapes
:*
T0*
Tshape0*+
_class!
loc:@loss/dense_4_loss/Mean_3
�
;training/Adam/gradients/loss/dense_4_loss/Mean_3_grad/ShapeShapeloss/dense_4_loss/truediv*
T0*
out_type0*+
_class!
loc:@loss/dense_4_loss/Mean_3*
_output_shapes
:
�
:training/Adam/gradients/loss/dense_4_loss/Mean_3_grad/TileTile=training/Adam/gradients/loss/dense_4_loss/Mean_3_grad/Reshape;training/Adam/gradients/loss/dense_4_loss/Mean_3_grad/Shape*

Tmultiples0*
T0*+
_class!
loc:@loss/dense_4_loss/Mean_3*#
_output_shapes
:���������
�
=training/Adam/gradients/loss/dense_4_loss/Mean_3_grad/Shape_1Shapeloss/dense_4_loss/truediv*
out_type0*+
_class!
loc:@loss/dense_4_loss/Mean_3*
_output_shapes
:*
T0
�
=training/Adam/gradients/loss/dense_4_loss/Mean_3_grad/Shape_2Const*
valueB *+
_class!
loc:@loss/dense_4_loss/Mean_3*
dtype0*
_output_shapes
: 
�
;training/Adam/gradients/loss/dense_4_loss/Mean_3_grad/ConstConst*
valueB: *+
_class!
loc:@loss/dense_4_loss/Mean_3*
dtype0*
_output_shapes
:
�
:training/Adam/gradients/loss/dense_4_loss/Mean_3_grad/ProdProd=training/Adam/gradients/loss/dense_4_loss/Mean_3_grad/Shape_1;training/Adam/gradients/loss/dense_4_loss/Mean_3_grad/Const*
T0*+
_class!
loc:@loss/dense_4_loss/Mean_3*
_output_shapes
: *

Tidx0*
	keep_dims( 
�
=training/Adam/gradients/loss/dense_4_loss/Mean_3_grad/Const_1Const*
dtype0*
_output_shapes
:*
valueB: *+
_class!
loc:@loss/dense_4_loss/Mean_3
�
<training/Adam/gradients/loss/dense_4_loss/Mean_3_grad/Prod_1Prod=training/Adam/gradients/loss/dense_4_loss/Mean_3_grad/Shape_2=training/Adam/gradients/loss/dense_4_loss/Mean_3_grad/Const_1*
T0*+
_class!
loc:@loss/dense_4_loss/Mean_3*
_output_shapes
: *

Tidx0*
	keep_dims( 
�
?training/Adam/gradients/loss/dense_4_loss/Mean_3_grad/Maximum/yConst*
value	B :*+
_class!
loc:@loss/dense_4_loss/Mean_3*
dtype0*
_output_shapes
: 
�
=training/Adam/gradients/loss/dense_4_loss/Mean_3_grad/MaximumMaximum<training/Adam/gradients/loss/dense_4_loss/Mean_3_grad/Prod_1?training/Adam/gradients/loss/dense_4_loss/Mean_3_grad/Maximum/y*
T0*+
_class!
loc:@loss/dense_4_loss/Mean_3*
_output_shapes
: 
�
>training/Adam/gradients/loss/dense_4_loss/Mean_3_grad/floordivFloorDiv:training/Adam/gradients/loss/dense_4_loss/Mean_3_grad/Prod=training/Adam/gradients/loss/dense_4_loss/Mean_3_grad/Maximum*
T0*+
_class!
loc:@loss/dense_4_loss/Mean_3*
_output_shapes
: 
�
:training/Adam/gradients/loss/dense_4_loss/Mean_3_grad/CastCast>training/Adam/gradients/loss/dense_4_loss/Mean_3_grad/floordiv*+
_class!
loc:@loss/dense_4_loss/Mean_3*
Truncate( *
_output_shapes
: *

DstT0*

SrcT0
�
=training/Adam/gradients/loss/dense_4_loss/Mean_3_grad/truedivRealDiv:training/Adam/gradients/loss/dense_4_loss/Mean_3_grad/Tile:training/Adam/gradients/loss/dense_4_loss/Mean_3_grad/Cast*
T0*+
_class!
loc:@loss/dense_4_loss/Mean_3*#
_output_shapes
:���������
�
Ktraining/Adam/gradients/dense_1/activity_regularizer/Sum_grad/Reshape/shapeConst*
valueB"      *3
_class)
'%loc:@dense_1/activity_regularizer/Sum*
dtype0*
_output_shapes
:
�
Etraining/Adam/gradients/dense_1/activity_regularizer/Sum_grad/ReshapeReshapetraining/Adam/gradients/FillKtraining/Adam/gradients/dense_1/activity_regularizer/Sum_grad/Reshape/shape*
T0*
Tshape0*3
_class)
'%loc:@dense_1/activity_regularizer/Sum*
_output_shapes

:
�
Ctraining/Adam/gradients/dense_1/activity_regularizer/Sum_grad/ShapeShape dense_1/activity_regularizer/mul*
_output_shapes
:*
T0*
out_type0*3
_class)
'%loc:@dense_1/activity_regularizer/Sum
�
Btraining/Adam/gradients/dense_1/activity_regularizer/Sum_grad/TileTileEtraining/Adam/gradients/dense_1/activity_regularizer/Sum_grad/ReshapeCtraining/Adam/gradients/dense_1/activity_regularizer/Sum_grad/Shape*

Tmultiples0*
T0*3
_class)
'%loc:@dense_1/activity_regularizer/Sum*'
_output_shapes
:���������

�
<training/Adam/gradients/loss/dense_4_loss/truediv_grad/ShapeShapeloss/dense_4_loss/mul*
T0*
out_type0*,
_class"
 loc:@loss/dense_4_loss/truediv*
_output_shapes
:
�
>training/Adam/gradients/loss/dense_4_loss/truediv_grad/Shape_1Const*
valueB *,
_class"
 loc:@loss/dense_4_loss/truediv*
dtype0*
_output_shapes
: 
�
Ltraining/Adam/gradients/loss/dense_4_loss/truediv_grad/BroadcastGradientArgsBroadcastGradientArgs<training/Adam/gradients/loss/dense_4_loss/truediv_grad/Shape>training/Adam/gradients/loss/dense_4_loss/truediv_grad/Shape_1*
T0*,
_class"
 loc:@loss/dense_4_loss/truediv*2
_output_shapes 
:���������:���������
�
>training/Adam/gradients/loss/dense_4_loss/truediv_grad/RealDivRealDiv=training/Adam/gradients/loss/dense_4_loss/Mean_3_grad/truedivloss/dense_4_loss/Mean_2*#
_output_shapes
:���������*
T0*,
_class"
 loc:@loss/dense_4_loss/truediv
�
:training/Adam/gradients/loss/dense_4_loss/truediv_grad/SumSum>training/Adam/gradients/loss/dense_4_loss/truediv_grad/RealDivLtraining/Adam/gradients/loss/dense_4_loss/truediv_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*,
_class"
 loc:@loss/dense_4_loss/truediv*
_output_shapes
:
�
>training/Adam/gradients/loss/dense_4_loss/truediv_grad/ReshapeReshape:training/Adam/gradients/loss/dense_4_loss/truediv_grad/Sum<training/Adam/gradients/loss/dense_4_loss/truediv_grad/Shape*
T0*
Tshape0*,
_class"
 loc:@loss/dense_4_loss/truediv*#
_output_shapes
:���������
�
:training/Adam/gradients/loss/dense_4_loss/truediv_grad/NegNegloss/dense_4_loss/mul*,
_class"
 loc:@loss/dense_4_loss/truediv*#
_output_shapes
:���������*
T0
�
@training/Adam/gradients/loss/dense_4_loss/truediv_grad/RealDiv_1RealDiv:training/Adam/gradients/loss/dense_4_loss/truediv_grad/Negloss/dense_4_loss/Mean_2*#
_output_shapes
:���������*
T0*,
_class"
 loc:@loss/dense_4_loss/truediv
�
@training/Adam/gradients/loss/dense_4_loss/truediv_grad/RealDiv_2RealDiv@training/Adam/gradients/loss/dense_4_loss/truediv_grad/RealDiv_1loss/dense_4_loss/Mean_2*
T0*,
_class"
 loc:@loss/dense_4_loss/truediv*#
_output_shapes
:���������
�
:training/Adam/gradients/loss/dense_4_loss/truediv_grad/mulMul=training/Adam/gradients/loss/dense_4_loss/Mean_3_grad/truediv@training/Adam/gradients/loss/dense_4_loss/truediv_grad/RealDiv_2*#
_output_shapes
:���������*
T0*,
_class"
 loc:@loss/dense_4_loss/truediv
�
<training/Adam/gradients/loss/dense_4_loss/truediv_grad/Sum_1Sum:training/Adam/gradients/loss/dense_4_loss/truediv_grad/mulNtraining/Adam/gradients/loss/dense_4_loss/truediv_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0*,
_class"
 loc:@loss/dense_4_loss/truediv
�
@training/Adam/gradients/loss/dense_4_loss/truediv_grad/Reshape_1Reshape<training/Adam/gradients/loss/dense_4_loss/truediv_grad/Sum_1>training/Adam/gradients/loss/dense_4_loss/truediv_grad/Shape_1*
T0*
Tshape0*,
_class"
 loc:@loss/dense_4_loss/truediv*
_output_shapes
: 
�
Ctraining/Adam/gradients/dense_1/activity_regularizer/mul_grad/ShapeConst*
dtype0*
_output_shapes
: *
valueB *3
_class)
'%loc:@dense_1/activity_regularizer/mul
�
Etraining/Adam/gradients/dense_1/activity_regularizer/mul_grad/Shape_1Shape dense_1/activity_regularizer/Abs*
T0*
out_type0*3
_class)
'%loc:@dense_1/activity_regularizer/mul*
_output_shapes
:
�
Straining/Adam/gradients/dense_1/activity_regularizer/mul_grad/BroadcastGradientArgsBroadcastGradientArgsCtraining/Adam/gradients/dense_1/activity_regularizer/mul_grad/ShapeEtraining/Adam/gradients/dense_1/activity_regularizer/mul_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0*3
_class)
'%loc:@dense_1/activity_regularizer/mul
�
Atraining/Adam/gradients/dense_1/activity_regularizer/mul_grad/MulMulBtraining/Adam/gradients/dense_1/activity_regularizer/Sum_grad/Tile dense_1/activity_regularizer/Abs*'
_output_shapes
:���������
*
T0*3
_class)
'%loc:@dense_1/activity_regularizer/mul
�
Atraining/Adam/gradients/dense_1/activity_regularizer/mul_grad/SumSumAtraining/Adam/gradients/dense_1/activity_regularizer/mul_grad/MulStraining/Adam/gradients/dense_1/activity_regularizer/mul_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0*3
_class)
'%loc:@dense_1/activity_regularizer/mul
�
Etraining/Adam/gradients/dense_1/activity_regularizer/mul_grad/ReshapeReshapeAtraining/Adam/gradients/dense_1/activity_regularizer/mul_grad/SumCtraining/Adam/gradients/dense_1/activity_regularizer/mul_grad/Shape*
_output_shapes
: *
T0*
Tshape0*3
_class)
'%loc:@dense_1/activity_regularizer/mul
�
Ctraining/Adam/gradients/dense_1/activity_regularizer/mul_grad/Mul_1Mul"dense_1/activity_regularizer/mul/xBtraining/Adam/gradients/dense_1/activity_regularizer/Sum_grad/Tile*'
_output_shapes
:���������
*
T0*3
_class)
'%loc:@dense_1/activity_regularizer/mul
�
Ctraining/Adam/gradients/dense_1/activity_regularizer/mul_grad/Sum_1SumCtraining/Adam/gradients/dense_1/activity_regularizer/mul_grad/Mul_1Utraining/Adam/gradients/dense_1/activity_regularizer/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0*3
_class)
'%loc:@dense_1/activity_regularizer/mul
�
Gtraining/Adam/gradients/dense_1/activity_regularizer/mul_grad/Reshape_1ReshapeCtraining/Adam/gradients/dense_1/activity_regularizer/mul_grad/Sum_1Etraining/Adam/gradients/dense_1/activity_regularizer/mul_grad/Shape_1*
T0*
Tshape0*3
_class)
'%loc:@dense_1/activity_regularizer/mul*'
_output_shapes
:���������

�
8training/Adam/gradients/loss/dense_4_loss/mul_grad/ShapeShapeloss/dense_4_loss/Mean_1*
_output_shapes
:*
T0*
out_type0*(
_class
loc:@loss/dense_4_loss/mul
�
:training/Adam/gradients/loss/dense_4_loss/mul_grad/Shape_1Shapedense_4_sample_weights*
T0*
out_type0*(
_class
loc:@loss/dense_4_loss/mul*
_output_shapes
:
�
Htraining/Adam/gradients/loss/dense_4_loss/mul_grad/BroadcastGradientArgsBroadcastGradientArgs8training/Adam/gradients/loss/dense_4_loss/mul_grad/Shape:training/Adam/gradients/loss/dense_4_loss/mul_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0*(
_class
loc:@loss/dense_4_loss/mul
�
6training/Adam/gradients/loss/dense_4_loss/mul_grad/MulMul>training/Adam/gradients/loss/dense_4_loss/truediv_grad/Reshapedense_4_sample_weights*#
_output_shapes
:���������*
T0*(
_class
loc:@loss/dense_4_loss/mul
�
6training/Adam/gradients/loss/dense_4_loss/mul_grad/SumSum6training/Adam/gradients/loss/dense_4_loss/mul_grad/MulHtraining/Adam/gradients/loss/dense_4_loss/mul_grad/BroadcastGradientArgs*
T0*(
_class
loc:@loss/dense_4_loss/mul*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
:training/Adam/gradients/loss/dense_4_loss/mul_grad/ReshapeReshape6training/Adam/gradients/loss/dense_4_loss/mul_grad/Sum8training/Adam/gradients/loss/dense_4_loss/mul_grad/Shape*
T0*
Tshape0*(
_class
loc:@loss/dense_4_loss/mul*#
_output_shapes
:���������
�
8training/Adam/gradients/loss/dense_4_loss/mul_grad/Mul_1Mulloss/dense_4_loss/Mean_1>training/Adam/gradients/loss/dense_4_loss/truediv_grad/Reshape*#
_output_shapes
:���������*
T0*(
_class
loc:@loss/dense_4_loss/mul
�
8training/Adam/gradients/loss/dense_4_loss/mul_grad/Sum_1Sum8training/Adam/gradients/loss/dense_4_loss/mul_grad/Mul_1Jtraining/Adam/gradients/loss/dense_4_loss/mul_grad/BroadcastGradientArgs:1*
T0*(
_class
loc:@loss/dense_4_loss/mul*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
<training/Adam/gradients/loss/dense_4_loss/mul_grad/Reshape_1Reshape8training/Adam/gradients/loss/dense_4_loss/mul_grad/Sum_1:training/Adam/gradients/loss/dense_4_loss/mul_grad/Shape_1*
T0*
Tshape0*(
_class
loc:@loss/dense_4_loss/mul*#
_output_shapes
:���������
�
Btraining/Adam/gradients/dense_1/activity_regularizer/Abs_grad/SignSigndense_1/Tanh*
T0*3
_class)
'%loc:@dense_1/activity_regularizer/Abs*'
_output_shapes
:���������

�
Atraining/Adam/gradients/dense_1/activity_regularizer/Abs_grad/mulMulGtraining/Adam/gradients/dense_1/activity_regularizer/mul_grad/Reshape_1Btraining/Adam/gradients/dense_1/activity_regularizer/Abs_grad/Sign*
T0*3
_class)
'%loc:@dense_1/activity_regularizer/Abs*'
_output_shapes
:���������

�
;training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/ShapeShapeloss/dense_4_loss/Mean*
T0*
out_type0*+
_class!
loc:@loss/dense_4_loss/Mean_1*
_output_shapes
:
�
:training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/SizeConst*
value	B :*+
_class!
loc:@loss/dense_4_loss/Mean_1*
dtype0*
_output_shapes
: 
�
9training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/addAdd*loss/dense_4_loss/Mean_1/reduction_indices:training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Size*
T0*+
_class!
loc:@loss/dense_4_loss/Mean_1*
_output_shapes
: 
�
9training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/modFloorMod9training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/add:training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Size*
T0*+
_class!
loc:@loss/dense_4_loss/Mean_1*
_output_shapes
: 
�
=training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB: *+
_class!
loc:@loss/dense_4_loss/Mean_1
�
Atraining/Adam/gradients/loss/dense_4_loss/Mean_1_grad/range/startConst*
_output_shapes
: *
value	B : *+
_class!
loc:@loss/dense_4_loss/Mean_1*
dtype0
�
Atraining/Adam/gradients/loss/dense_4_loss/Mean_1_grad/range/deltaConst*
value	B :*+
_class!
loc:@loss/dense_4_loss/Mean_1*
dtype0*
_output_shapes
: 
�
;training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/rangeRangeAtraining/Adam/gradients/loss/dense_4_loss/Mean_1_grad/range/start:training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/SizeAtraining/Adam/gradients/loss/dense_4_loss/Mean_1_grad/range/delta*+
_class!
loc:@loss/dense_4_loss/Mean_1*
_output_shapes
:*

Tidx0
�
@training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Fill/valueConst*
value	B :*+
_class!
loc:@loss/dense_4_loss/Mean_1*
dtype0*
_output_shapes
: 
�
:training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/FillFill=training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Shape_1@training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Fill/value*

index_type0*+
_class!
loc:@loss/dense_4_loss/Mean_1*
_output_shapes
: *
T0
�
Ctraining/Adam/gradients/loss/dense_4_loss/Mean_1_grad/DynamicStitchDynamicStitch;training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/range9training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/mod;training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Shape:training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Fill*
T0*+
_class!
loc:@loss/dense_4_loss/Mean_1*
N*
_output_shapes
:
�
?training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Maximum/yConst*
value	B :*+
_class!
loc:@loss/dense_4_loss/Mean_1*
dtype0*
_output_shapes
: 
�
=training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/MaximumMaximumCtraining/Adam/gradients/loss/dense_4_loss/Mean_1_grad/DynamicStitch?training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Maximum/y*
_output_shapes
:*
T0*+
_class!
loc:@loss/dense_4_loss/Mean_1
�
>training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/floordivFloorDiv;training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Shape=training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Maximum*+
_class!
loc:@loss/dense_4_loss/Mean_1*
_output_shapes
:*
T0
�
=training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/ReshapeReshape:training/Adam/gradients/loss/dense_4_loss/mul_grad/ReshapeCtraining/Adam/gradients/loss/dense_4_loss/Mean_1_grad/DynamicStitch*#
_output_shapes
:���������*
T0*
Tshape0*+
_class!
loc:@loss/dense_4_loss/Mean_1
�
:training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/TileTile=training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Reshape>training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/floordiv*#
_output_shapes
:���������*

Tmultiples0*
T0*+
_class!
loc:@loss/dense_4_loss/Mean_1
�
=training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Shape_2Shapeloss/dense_4_loss/Mean*
T0*
out_type0*+
_class!
loc:@loss/dense_4_loss/Mean_1*
_output_shapes
:
�
=training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Shape_3Shapeloss/dense_4_loss/Mean_1*
T0*
out_type0*+
_class!
loc:@loss/dense_4_loss/Mean_1*
_output_shapes
:
�
;training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/ConstConst*
valueB: *+
_class!
loc:@loss/dense_4_loss/Mean_1*
dtype0*
_output_shapes
:
�
:training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/ProdProd=training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Shape_2;training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Const*

Tidx0*
	keep_dims( *
T0*+
_class!
loc:@loss/dense_4_loss/Mean_1*
_output_shapes
: 
�
=training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Const_1Const*
dtype0*
_output_shapes
:*
valueB: *+
_class!
loc:@loss/dense_4_loss/Mean_1
�
<training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Prod_1Prod=training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Shape_3=training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Const_1*
T0*+
_class!
loc:@loss/dense_4_loss/Mean_1*
_output_shapes
: *

Tidx0*
	keep_dims( 
�
Atraining/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Maximum_1/yConst*
value	B :*+
_class!
loc:@loss/dense_4_loss/Mean_1*
dtype0*
_output_shapes
: 
�
?training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Maximum_1Maximum<training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Prod_1Atraining/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Maximum_1/y*
T0*+
_class!
loc:@loss/dense_4_loss/Mean_1*
_output_shapes
: 
�
@training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/floordiv_1FloorDiv:training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Prod?training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Maximum_1*
T0*+
_class!
loc:@loss/dense_4_loss/Mean_1*
_output_shapes
: 
�
:training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/CastCast@training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/floordiv_1*

SrcT0*+
_class!
loc:@loss/dense_4_loss/Mean_1*
Truncate( *
_output_shapes
: *

DstT0
�
=training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/truedivRealDiv:training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Tile:training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Cast*
T0*+
_class!
loc:@loss/dense_4_loss/Mean_1*#
_output_shapes
:���������
�
9training/Adam/gradients/loss/dense_4_loss/Mean_grad/ShapeShapeloss/dense_4_loss/Square*
_output_shapes
:*
T0*
out_type0*)
_class
loc:@loss/dense_4_loss/Mean
�
8training/Adam/gradients/loss/dense_4_loss/Mean_grad/SizeConst*
value	B :*)
_class
loc:@loss/dense_4_loss/Mean*
dtype0*
_output_shapes
: 
�
7training/Adam/gradients/loss/dense_4_loss/Mean_grad/addAdd(loss/dense_4_loss/Mean/reduction_indices8training/Adam/gradients/loss/dense_4_loss/Mean_grad/Size*
T0*)
_class
loc:@loss/dense_4_loss/Mean*
_output_shapes
: 
�
7training/Adam/gradients/loss/dense_4_loss/Mean_grad/modFloorMod7training/Adam/gradients/loss/dense_4_loss/Mean_grad/add8training/Adam/gradients/loss/dense_4_loss/Mean_grad/Size*
T0*)
_class
loc:@loss/dense_4_loss/Mean*
_output_shapes
: 
�
;training/Adam/gradients/loss/dense_4_loss/Mean_grad/Shape_1Const*
valueB *)
_class
loc:@loss/dense_4_loss/Mean*
dtype0*
_output_shapes
: 
�
?training/Adam/gradients/loss/dense_4_loss/Mean_grad/range/startConst*
_output_shapes
: *
value	B : *)
_class
loc:@loss/dense_4_loss/Mean*
dtype0
�
?training/Adam/gradients/loss/dense_4_loss/Mean_grad/range/deltaConst*
value	B :*)
_class
loc:@loss/dense_4_loss/Mean*
dtype0*
_output_shapes
: 
�
9training/Adam/gradients/loss/dense_4_loss/Mean_grad/rangeRange?training/Adam/gradients/loss/dense_4_loss/Mean_grad/range/start8training/Adam/gradients/loss/dense_4_loss/Mean_grad/Size?training/Adam/gradients/loss/dense_4_loss/Mean_grad/range/delta*

Tidx0*)
_class
loc:@loss/dense_4_loss/Mean*
_output_shapes
:
�
>training/Adam/gradients/loss/dense_4_loss/Mean_grad/Fill/valueConst*
value	B :*)
_class
loc:@loss/dense_4_loss/Mean*
dtype0*
_output_shapes
: 
�
8training/Adam/gradients/loss/dense_4_loss/Mean_grad/FillFill;training/Adam/gradients/loss/dense_4_loss/Mean_grad/Shape_1>training/Adam/gradients/loss/dense_4_loss/Mean_grad/Fill/value*
T0*

index_type0*)
_class
loc:@loss/dense_4_loss/Mean*
_output_shapes
: 
�
Atraining/Adam/gradients/loss/dense_4_loss/Mean_grad/DynamicStitchDynamicStitch9training/Adam/gradients/loss/dense_4_loss/Mean_grad/range7training/Adam/gradients/loss/dense_4_loss/Mean_grad/mod9training/Adam/gradients/loss/dense_4_loss/Mean_grad/Shape8training/Adam/gradients/loss/dense_4_loss/Mean_grad/Fill*
T0*)
_class
loc:@loss/dense_4_loss/Mean*
N*
_output_shapes
:
�
=training/Adam/gradients/loss/dense_4_loss/Mean_grad/Maximum/yConst*
_output_shapes
: *
value	B :*)
_class
loc:@loss/dense_4_loss/Mean*
dtype0
�
;training/Adam/gradients/loss/dense_4_loss/Mean_grad/MaximumMaximumAtraining/Adam/gradients/loss/dense_4_loss/Mean_grad/DynamicStitch=training/Adam/gradients/loss/dense_4_loss/Mean_grad/Maximum/y*
T0*)
_class
loc:@loss/dense_4_loss/Mean*
_output_shapes
:
�
<training/Adam/gradients/loss/dense_4_loss/Mean_grad/floordivFloorDiv9training/Adam/gradients/loss/dense_4_loss/Mean_grad/Shape;training/Adam/gradients/loss/dense_4_loss/Mean_grad/Maximum*
_output_shapes
:*
T0*)
_class
loc:@loss/dense_4_loss/Mean
�
;training/Adam/gradients/loss/dense_4_loss/Mean_grad/ReshapeReshape=training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/truedivAtraining/Adam/gradients/loss/dense_4_loss/Mean_grad/DynamicStitch*
T0*
Tshape0*)
_class
loc:@loss/dense_4_loss/Mean*0
_output_shapes
:������������������
�
8training/Adam/gradients/loss/dense_4_loss/Mean_grad/TileTile;training/Adam/gradients/loss/dense_4_loss/Mean_grad/Reshape<training/Adam/gradients/loss/dense_4_loss/Mean_grad/floordiv*)
_class
loc:@loss/dense_4_loss/Mean*0
_output_shapes
:������������������*

Tmultiples0*
T0
�
;training/Adam/gradients/loss/dense_4_loss/Mean_grad/Shape_2Shapeloss/dense_4_loss/Square*
_output_shapes
:*
T0*
out_type0*)
_class
loc:@loss/dense_4_loss/Mean
�
;training/Adam/gradients/loss/dense_4_loss/Mean_grad/Shape_3Shapeloss/dense_4_loss/Mean*
T0*
out_type0*)
_class
loc:@loss/dense_4_loss/Mean*
_output_shapes
:
�
9training/Adam/gradients/loss/dense_4_loss/Mean_grad/ConstConst*
valueB: *)
_class
loc:@loss/dense_4_loss/Mean*
dtype0*
_output_shapes
:
�
8training/Adam/gradients/loss/dense_4_loss/Mean_grad/ProdProd;training/Adam/gradients/loss/dense_4_loss/Mean_grad/Shape_29training/Adam/gradients/loss/dense_4_loss/Mean_grad/Const*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0*)
_class
loc:@loss/dense_4_loss/Mean
�
;training/Adam/gradients/loss/dense_4_loss/Mean_grad/Const_1Const*
dtype0*
_output_shapes
:*
valueB: *)
_class
loc:@loss/dense_4_loss/Mean
�
:training/Adam/gradients/loss/dense_4_loss/Mean_grad/Prod_1Prod;training/Adam/gradients/loss/dense_4_loss/Mean_grad/Shape_3;training/Adam/gradients/loss/dense_4_loss/Mean_grad/Const_1*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0*)
_class
loc:@loss/dense_4_loss/Mean
�
?training/Adam/gradients/loss/dense_4_loss/Mean_grad/Maximum_1/yConst*
value	B :*)
_class
loc:@loss/dense_4_loss/Mean*
dtype0*
_output_shapes
: 
�
=training/Adam/gradients/loss/dense_4_loss/Mean_grad/Maximum_1Maximum:training/Adam/gradients/loss/dense_4_loss/Mean_grad/Prod_1?training/Adam/gradients/loss/dense_4_loss/Mean_grad/Maximum_1/y*
_output_shapes
: *
T0*)
_class
loc:@loss/dense_4_loss/Mean
�
>training/Adam/gradients/loss/dense_4_loss/Mean_grad/floordiv_1FloorDiv8training/Adam/gradients/loss/dense_4_loss/Mean_grad/Prod=training/Adam/gradients/loss/dense_4_loss/Mean_grad/Maximum_1*
T0*)
_class
loc:@loss/dense_4_loss/Mean*
_output_shapes
: 
�
8training/Adam/gradients/loss/dense_4_loss/Mean_grad/CastCast>training/Adam/gradients/loss/dense_4_loss/Mean_grad/floordiv_1*

SrcT0*)
_class
loc:@loss/dense_4_loss/Mean*
Truncate( *
_output_shapes
: *

DstT0
�
;training/Adam/gradients/loss/dense_4_loss/Mean_grad/truedivRealDiv8training/Adam/gradients/loss/dense_4_loss/Mean_grad/Tile8training/Adam/gradients/loss/dense_4_loss/Mean_grad/Cast*'
_output_shapes
:���������+*
T0*)
_class
loc:@loss/dense_4_loss/Mean
�
;training/Adam/gradients/loss/dense_4_loss/Square_grad/ConstConst<^training/Adam/gradients/loss/dense_4_loss/Mean_grad/truediv*
valueB
 *   @*+
_class!
loc:@loss/dense_4_loss/Square*
dtype0*
_output_shapes
: 
�
9training/Adam/gradients/loss/dense_4_loss/Square_grad/MulMulloss/dense_4_loss/sub;training/Adam/gradients/loss/dense_4_loss/Square_grad/Const*
T0*+
_class!
loc:@loss/dense_4_loss/Square*'
_output_shapes
:���������+
�
;training/Adam/gradients/loss/dense_4_loss/Square_grad/Mul_1Mul;training/Adam/gradients/loss/dense_4_loss/Mean_grad/truediv9training/Adam/gradients/loss/dense_4_loss/Square_grad/Mul*'
_output_shapes
:���������+*
T0*+
_class!
loc:@loss/dense_4_loss/Square
�
8training/Adam/gradients/loss/dense_4_loss/sub_grad/ShapeShapedense_4/Relu*
T0*
out_type0*(
_class
loc:@loss/dense_4_loss/sub*
_output_shapes
:
�
:training/Adam/gradients/loss/dense_4_loss/sub_grad/Shape_1Shapedense_4_target*
T0*
out_type0*(
_class
loc:@loss/dense_4_loss/sub*
_output_shapes
:
�
Htraining/Adam/gradients/loss/dense_4_loss/sub_grad/BroadcastGradientArgsBroadcastGradientArgs8training/Adam/gradients/loss/dense_4_loss/sub_grad/Shape:training/Adam/gradients/loss/dense_4_loss/sub_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0*(
_class
loc:@loss/dense_4_loss/sub
�
6training/Adam/gradients/loss/dense_4_loss/sub_grad/SumSum;training/Adam/gradients/loss/dense_4_loss/Square_grad/Mul_1Htraining/Adam/gradients/loss/dense_4_loss/sub_grad/BroadcastGradientArgs*
T0*(
_class
loc:@loss/dense_4_loss/sub*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
:training/Adam/gradients/loss/dense_4_loss/sub_grad/ReshapeReshape6training/Adam/gradients/loss/dense_4_loss/sub_grad/Sum8training/Adam/gradients/loss/dense_4_loss/sub_grad/Shape*'
_output_shapes
:���������+*
T0*
Tshape0*(
_class
loc:@loss/dense_4_loss/sub
�
8training/Adam/gradients/loss/dense_4_loss/sub_grad/Sum_1Sum;training/Adam/gradients/loss/dense_4_loss/Square_grad/Mul_1Jtraining/Adam/gradients/loss/dense_4_loss/sub_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0*(
_class
loc:@loss/dense_4_loss/sub
�
6training/Adam/gradients/loss/dense_4_loss/sub_grad/NegNeg8training/Adam/gradients/loss/dense_4_loss/sub_grad/Sum_1*(
_class
loc:@loss/dense_4_loss/sub*
_output_shapes
:*
T0
�
<training/Adam/gradients/loss/dense_4_loss/sub_grad/Reshape_1Reshape6training/Adam/gradients/loss/dense_4_loss/sub_grad/Neg:training/Adam/gradients/loss/dense_4_loss/sub_grad/Shape_1*
T0*
Tshape0*(
_class
loc:@loss/dense_4_loss/sub*0
_output_shapes
:������������������
�
2training/Adam/gradients/dense_4/Relu_grad/ReluGradReluGrad:training/Adam/gradients/loss/dense_4_loss/sub_grad/Reshapedense_4/Relu*'
_output_shapes
:���������+*
T0*
_class
loc:@dense_4/Relu
�
8training/Adam/gradients/dense_4/BiasAdd_grad/BiasAddGradBiasAddGrad2training/Adam/gradients/dense_4/Relu_grad/ReluGrad*
T0*"
_class
loc:@dense_4/BiasAdd*
data_formatNHWC*
_output_shapes
:+
�
2training/Adam/gradients/dense_4/MatMul_grad/MatMulMatMul2training/Adam/gradients/dense_4/Relu_grad/ReluGraddense_4/kernel/read*'
_output_shapes
:���������*
transpose_a( *
transpose_b(*
T0*!
_class
loc:@dense_4/MatMul
�
4training/Adam/gradients/dense_4/MatMul_grad/MatMul_1MatMuldense_3/Tanh2training/Adam/gradients/dense_4/Relu_grad/ReluGrad*
T0*!
_class
loc:@dense_4/MatMul*
_output_shapes

:+*
transpose_a(*
transpose_b( 
�
2training/Adam/gradients/dense_3/Tanh_grad/TanhGradTanhGraddense_3/Tanh2training/Adam/gradients/dense_4/MatMul_grad/MatMul*
_class
loc:@dense_3/Tanh*'
_output_shapes
:���������*
T0
�
8training/Adam/gradients/dense_3/BiasAdd_grad/BiasAddGradBiasAddGrad2training/Adam/gradients/dense_3/Tanh_grad/TanhGrad*
data_formatNHWC*
_output_shapes
:*
T0*"
_class
loc:@dense_3/BiasAdd
�
2training/Adam/gradients/dense_3/MatMul_grad/MatMulMatMul2training/Adam/gradients/dense_3/Tanh_grad/TanhGraddense_3/kernel/read*
T0*!
_class
loc:@dense_3/MatMul*'
_output_shapes
:���������*
transpose_a( *
transpose_b(
�
4training/Adam/gradients/dense_3/MatMul_grad/MatMul_1MatMuldense_2/Relu2training/Adam/gradients/dense_3/Tanh_grad/TanhGrad*
T0*!
_class
loc:@dense_3/MatMul*
_output_shapes

:*
transpose_a(*
transpose_b( 
�
2training/Adam/gradients/dense_2/Relu_grad/ReluGradReluGrad2training/Adam/gradients/dense_3/MatMul_grad/MatMuldense_2/Relu*
T0*
_class
loc:@dense_2/Relu*'
_output_shapes
:���������
�
8training/Adam/gradients/dense_2/BiasAdd_grad/BiasAddGradBiasAddGrad2training/Adam/gradients/dense_2/Relu_grad/ReluGrad*
T0*"
_class
loc:@dense_2/BiasAdd*
data_formatNHWC*
_output_shapes
:
�
2training/Adam/gradients/dense_2/MatMul_grad/MatMulMatMul2training/Adam/gradients/dense_2/Relu_grad/ReluGraddense_2/kernel/read*
T0*!
_class
loc:@dense_2/MatMul*'
_output_shapes
:���������
*
transpose_a( *
transpose_b(
�
4training/Adam/gradients/dense_2/MatMul_grad/MatMul_1MatMuldense_1/Tanh2training/Adam/gradients/dense_2/Relu_grad/ReluGrad*
_output_shapes

:
*
transpose_a(*
transpose_b( *
T0*!
_class
loc:@dense_2/MatMul
�
training/Adam/gradients/AddNAddNAtraining/Adam/gradients/dense_1/activity_regularizer/Abs_grad/mul2training/Adam/gradients/dense_2/MatMul_grad/MatMul*3
_class)
'%loc:@dense_1/activity_regularizer/Abs*
N*'
_output_shapes
:���������
*
T0
�
2training/Adam/gradients/dense_1/Tanh_grad/TanhGradTanhGraddense_1/Tanhtraining/Adam/gradients/AddN*'
_output_shapes
:���������
*
T0*
_class
loc:@dense_1/Tanh
�
8training/Adam/gradients/dense_1/BiasAdd_grad/BiasAddGradBiasAddGrad2training/Adam/gradients/dense_1/Tanh_grad/TanhGrad*
_output_shapes
:
*
T0*"
_class
loc:@dense_1/BiasAdd*
data_formatNHWC
�
2training/Adam/gradients/dense_1/MatMul_grad/MatMulMatMul2training/Adam/gradients/dense_1/Tanh_grad/TanhGraddense_1/kernel/read*
T0*!
_class
loc:@dense_1/MatMul*'
_output_shapes
:���������+*
transpose_a( *
transpose_b(
�
4training/Adam/gradients/dense_1/MatMul_grad/MatMul_1MatMulinput_12training/Adam/gradients/dense_1/Tanh_grad/TanhGrad*
T0*!
_class
loc:@dense_1/MatMul*
_output_shapes

:+
*
transpose_a(*
transpose_b( 
_
training/Adam/AssignAdd/valueConst*
value	B	 R*
dtype0	*
_output_shapes
: 
�
training/Adam/AssignAdd	AssignAddAdam/iterationstraining/Adam/AssignAdd/value*
T0	*"
_class
loc:@Adam/iterations*
_output_shapes
: *
use_locking( 
p
training/Adam/CastCastAdam/iterations/read*
Truncate( *
_output_shapes
: *

DstT0*

SrcT0	
X
training/Adam/add/yConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
b
training/Adam/addAddtraining/Adam/Casttraining/Adam/add/y*
T0*
_output_shapes
: 
^
training/Adam/PowPowAdam/beta_2/readtraining/Adam/add*
T0*
_output_shapes
: 
X
training/Adam/sub/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
a
training/Adam/subSubtraining/Adam/sub/xtraining/Adam/Pow*
_output_shapes
: *
T0
X
training/Adam/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Z
training/Adam/Const_1Const*
valueB
 *  �*
dtype0*
_output_shapes
: 
y
#training/Adam/clip_by_value/MinimumMinimumtraining/Adam/subtraining/Adam/Const_1*
T0*
_output_shapes
: 
�
training/Adam/clip_by_valueMaximum#training/Adam/clip_by_value/Minimumtraining/Adam/Const*
_output_shapes
: *
T0
X
training/Adam/SqrtSqrttraining/Adam/clip_by_value*
_output_shapes
: *
T0
`
training/Adam/Pow_1PowAdam/beta_1/readtraining/Adam/add*
_output_shapes
: *
T0
Z
training/Adam/sub_1/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
g
training/Adam/sub_1Subtraining/Adam/sub_1/xtraining/Adam/Pow_1*
T0*
_output_shapes
: 
j
training/Adam/truedivRealDivtraining/Adam/Sqrttraining/Adam/sub_1*
T0*
_output_shapes
: 
^
training/Adam/mulMulAdam/lr/readtraining/Adam/truediv*
_output_shapes
: *
T0
h
training/Adam/zerosConst*
valueB+
*    *
dtype0*
_output_shapes

:+

�
training/Adam/Variable
VariableV2*
dtype0*
_output_shapes

:+
*
	container *
shape
:+
*
shared_name 
�
training/Adam/Variable/AssignAssigntraining/Adam/Variabletraining/Adam/zeros*
use_locking(*
T0*)
_class
loc:@training/Adam/Variable*
validate_shape(*
_output_shapes

:+

�
training/Adam/Variable/readIdentitytraining/Adam/Variable*
T0*)
_class
loc:@training/Adam/Variable*
_output_shapes

:+

b
training/Adam/zeros_1Const*
valueB
*    *
dtype0*
_output_shapes
:

�
training/Adam/Variable_1
VariableV2*
dtype0*
_output_shapes
:
*
	container *
shape:
*
shared_name 
�
training/Adam/Variable_1/AssignAssigntraining/Adam/Variable_1training/Adam/zeros_1*+
_class!
loc:@training/Adam/Variable_1*
validate_shape(*
_output_shapes
:
*
use_locking(*
T0
�
training/Adam/Variable_1/readIdentitytraining/Adam/Variable_1*
_output_shapes
:
*
T0*+
_class!
loc:@training/Adam/Variable_1
j
training/Adam/zeros_2Const*
valueB
*    *
dtype0*
_output_shapes

:

�
training/Adam/Variable_2
VariableV2*
shared_name *
dtype0*
_output_shapes

:
*
	container *
shape
:

�
training/Adam/Variable_2/AssignAssigntraining/Adam/Variable_2training/Adam/zeros_2*
validate_shape(*
_output_shapes

:
*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_2
�
training/Adam/Variable_2/readIdentitytraining/Adam/Variable_2*
T0*+
_class!
loc:@training/Adam/Variable_2*
_output_shapes

:

b
training/Adam/zeros_3Const*
valueB*    *
dtype0*
_output_shapes
:
�
training/Adam/Variable_3
VariableV2*
dtype0*
_output_shapes
:*
	container *
shape:*
shared_name 
�
training/Adam/Variable_3/AssignAssigntraining/Adam/Variable_3training/Adam/zeros_3*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_3*
validate_shape(*
_output_shapes
:
�
training/Adam/Variable_3/readIdentitytraining/Adam/Variable_3*
T0*+
_class!
loc:@training/Adam/Variable_3*
_output_shapes
:
j
training/Adam/zeros_4Const*
dtype0*
_output_shapes

:*
valueB*    
�
training/Adam/Variable_4
VariableV2*
dtype0*
_output_shapes

:*
	container *
shape
:*
shared_name 
�
training/Adam/Variable_4/AssignAssigntraining/Adam/Variable_4training/Adam/zeros_4*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_4
�
training/Adam/Variable_4/readIdentitytraining/Adam/Variable_4*
T0*+
_class!
loc:@training/Adam/Variable_4*
_output_shapes

:
b
training/Adam/zeros_5Const*
valueB*    *
dtype0*
_output_shapes
:
�
training/Adam/Variable_5
VariableV2*
shape:*
shared_name *
dtype0*
_output_shapes
:*
	container 
�
training/Adam/Variable_5/AssignAssigntraining/Adam/Variable_5training/Adam/zeros_5*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_5
�
training/Adam/Variable_5/readIdentitytraining/Adam/Variable_5*
T0*+
_class!
loc:@training/Adam/Variable_5*
_output_shapes
:
j
training/Adam/zeros_6Const*
valueB+*    *
dtype0*
_output_shapes

:+
�
training/Adam/Variable_6
VariableV2*
_output_shapes

:+*
	container *
shape
:+*
shared_name *
dtype0
�
training/Adam/Variable_6/AssignAssigntraining/Adam/Variable_6training/Adam/zeros_6*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_6*
validate_shape(*
_output_shapes

:+
�
training/Adam/Variable_6/readIdentitytraining/Adam/Variable_6*
T0*+
_class!
loc:@training/Adam/Variable_6*
_output_shapes

:+
b
training/Adam/zeros_7Const*
dtype0*
_output_shapes
:+*
valueB+*    
�
training/Adam/Variable_7
VariableV2*
shared_name *
dtype0*
_output_shapes
:+*
	container *
shape:+
�
training/Adam/Variable_7/AssignAssigntraining/Adam/Variable_7training/Adam/zeros_7*
T0*+
_class!
loc:@training/Adam/Variable_7*
validate_shape(*
_output_shapes
:+*
use_locking(
�
training/Adam/Variable_7/readIdentitytraining/Adam/Variable_7*
_output_shapes
:+*
T0*+
_class!
loc:@training/Adam/Variable_7
j
training/Adam/zeros_8Const*
valueB+
*    *
dtype0*
_output_shapes

:+

�
training/Adam/Variable_8
VariableV2*
_output_shapes

:+
*
	container *
shape
:+
*
shared_name *
dtype0
�
training/Adam/Variable_8/AssignAssigntraining/Adam/Variable_8training/Adam/zeros_8*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_8*
validate_shape(*
_output_shapes

:+

�
training/Adam/Variable_8/readIdentitytraining/Adam/Variable_8*+
_class!
loc:@training/Adam/Variable_8*
_output_shapes

:+
*
T0
b
training/Adam/zeros_9Const*
valueB
*    *
dtype0*
_output_shapes
:

�
training/Adam/Variable_9
VariableV2*
dtype0*
_output_shapes
:
*
	container *
shape:
*
shared_name 
�
training/Adam/Variable_9/AssignAssigntraining/Adam/Variable_9training/Adam/zeros_9*
T0*+
_class!
loc:@training/Adam/Variable_9*
validate_shape(*
_output_shapes
:
*
use_locking(
�
training/Adam/Variable_9/readIdentitytraining/Adam/Variable_9*
T0*+
_class!
loc:@training/Adam/Variable_9*
_output_shapes
:

k
training/Adam/zeros_10Const*
valueB
*    *
dtype0*
_output_shapes

:

�
training/Adam/Variable_10
VariableV2*
shape
:
*
shared_name *
dtype0*
_output_shapes

:
*
	container 
�
 training/Adam/Variable_10/AssignAssigntraining/Adam/Variable_10training/Adam/zeros_10*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_10*
validate_shape(*
_output_shapes

:

�
training/Adam/Variable_10/readIdentitytraining/Adam/Variable_10*
_output_shapes

:
*
T0*,
_class"
 loc:@training/Adam/Variable_10
c
training/Adam/zeros_11Const*
valueB*    *
dtype0*
_output_shapes
:
�
training/Adam/Variable_11
VariableV2*
dtype0*
_output_shapes
:*
	container *
shape:*
shared_name 
�
 training/Adam/Variable_11/AssignAssigntraining/Adam/Variable_11training/Adam/zeros_11*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_11*
validate_shape(*
_output_shapes
:
�
training/Adam/Variable_11/readIdentitytraining/Adam/Variable_11*
_output_shapes
:*
T0*,
_class"
 loc:@training/Adam/Variable_11
k
training/Adam/zeros_12Const*
dtype0*
_output_shapes

:*
valueB*    
�
training/Adam/Variable_12
VariableV2*
shared_name *
dtype0*
_output_shapes

:*
	container *
shape
:
�
 training/Adam/Variable_12/AssignAssigntraining/Adam/Variable_12training/Adam/zeros_12*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_12*
validate_shape(*
_output_shapes

:
�
training/Adam/Variable_12/readIdentitytraining/Adam/Variable_12*
T0*,
_class"
 loc:@training/Adam/Variable_12*
_output_shapes

:
c
training/Adam/zeros_13Const*
_output_shapes
:*
valueB*    *
dtype0
�
training/Adam/Variable_13
VariableV2*
dtype0*
_output_shapes
:*
	container *
shape:*
shared_name 
�
 training/Adam/Variable_13/AssignAssigntraining/Adam/Variable_13training/Adam/zeros_13*
_output_shapes
:*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_13*
validate_shape(
�
training/Adam/Variable_13/readIdentitytraining/Adam/Variable_13*
T0*,
_class"
 loc:@training/Adam/Variable_13*
_output_shapes
:
k
training/Adam/zeros_14Const*
valueB+*    *
dtype0*
_output_shapes

:+
�
training/Adam/Variable_14
VariableV2*
shape
:+*
shared_name *
dtype0*
_output_shapes

:+*
	container 
�
 training/Adam/Variable_14/AssignAssigntraining/Adam/Variable_14training/Adam/zeros_14*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_14*
validate_shape(*
_output_shapes

:+
�
training/Adam/Variable_14/readIdentitytraining/Adam/Variable_14*
T0*,
_class"
 loc:@training/Adam/Variable_14*
_output_shapes

:+
c
training/Adam/zeros_15Const*
valueB+*    *
dtype0*
_output_shapes
:+
�
training/Adam/Variable_15
VariableV2*
dtype0*
_output_shapes
:+*
	container *
shape:+*
shared_name 
�
 training/Adam/Variable_15/AssignAssigntraining/Adam/Variable_15training/Adam/zeros_15*
_output_shapes
:+*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_15*
validate_shape(
�
training/Adam/Variable_15/readIdentitytraining/Adam/Variable_15*
T0*,
_class"
 loc:@training/Adam/Variable_15*
_output_shapes
:+
p
&training/Adam/zeros_16/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
a
training/Adam/zeros_16/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
training/Adam/zeros_16Fill&training/Adam/zeros_16/shape_as_tensortraining/Adam/zeros_16/Const*

index_type0*
_output_shapes
:*
T0
�
training/Adam/Variable_16
VariableV2*
shape:*
shared_name *
dtype0*
_output_shapes
:*
	container 
�
 training/Adam/Variable_16/AssignAssigntraining/Adam/Variable_16training/Adam/zeros_16*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_16
�
training/Adam/Variable_16/readIdentitytraining/Adam/Variable_16*
T0*,
_class"
 loc:@training/Adam/Variable_16*
_output_shapes
:
p
&training/Adam/zeros_17/shape_as_tensorConst*
dtype0*
_output_shapes
:*
valueB:
a
training/Adam/zeros_17/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
training/Adam/zeros_17Fill&training/Adam/zeros_17/shape_as_tensortraining/Adam/zeros_17/Const*
T0*

index_type0*
_output_shapes
:
�
training/Adam/Variable_17
VariableV2*
_output_shapes
:*
	container *
shape:*
shared_name *
dtype0
�
 training/Adam/Variable_17/AssignAssigntraining/Adam/Variable_17training/Adam/zeros_17*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_17*
validate_shape(*
_output_shapes
:
�
training/Adam/Variable_17/readIdentitytraining/Adam/Variable_17*
T0*,
_class"
 loc:@training/Adam/Variable_17*
_output_shapes
:
p
&training/Adam/zeros_18/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
a
training/Adam/zeros_18/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
training/Adam/zeros_18Fill&training/Adam/zeros_18/shape_as_tensortraining/Adam/zeros_18/Const*
T0*

index_type0*
_output_shapes
:
�
training/Adam/Variable_18
VariableV2*
dtype0*
_output_shapes
:*
	container *
shape:*
shared_name 
�
 training/Adam/Variable_18/AssignAssigntraining/Adam/Variable_18training/Adam/zeros_18*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_18*
validate_shape(*
_output_shapes
:
�
training/Adam/Variable_18/readIdentitytraining/Adam/Variable_18*
_output_shapes
:*
T0*,
_class"
 loc:@training/Adam/Variable_18
p
&training/Adam/zeros_19/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
a
training/Adam/zeros_19/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
training/Adam/zeros_19Fill&training/Adam/zeros_19/shape_as_tensortraining/Adam/zeros_19/Const*
T0*

index_type0*
_output_shapes
:
�
training/Adam/Variable_19
VariableV2*
shared_name *
dtype0*
_output_shapes
:*
	container *
shape:
�
 training/Adam/Variable_19/AssignAssigntraining/Adam/Variable_19training/Adam/zeros_19*,
_class"
 loc:@training/Adam/Variable_19*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
�
training/Adam/Variable_19/readIdentitytraining/Adam/Variable_19*
T0*,
_class"
 loc:@training/Adam/Variable_19*
_output_shapes
:
p
&training/Adam/zeros_20/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
a
training/Adam/zeros_20/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
training/Adam/zeros_20Fill&training/Adam/zeros_20/shape_as_tensortraining/Adam/zeros_20/Const*
T0*

index_type0*
_output_shapes
:
�
training/Adam/Variable_20
VariableV2*
dtype0*
_output_shapes
:*
	container *
shape:*
shared_name 
�
 training/Adam/Variable_20/AssignAssigntraining/Adam/Variable_20training/Adam/zeros_20*
_output_shapes
:*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_20*
validate_shape(
�
training/Adam/Variable_20/readIdentitytraining/Adam/Variable_20*
T0*,
_class"
 loc:@training/Adam/Variable_20*
_output_shapes
:
p
&training/Adam/zeros_21/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
a
training/Adam/zeros_21/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
training/Adam/zeros_21Fill&training/Adam/zeros_21/shape_as_tensortraining/Adam/zeros_21/Const*

index_type0*
_output_shapes
:*
T0
�
training/Adam/Variable_21
VariableV2*
shared_name *
dtype0*
_output_shapes
:*
	container *
shape:
�
 training/Adam/Variable_21/AssignAssigntraining/Adam/Variable_21training/Adam/zeros_21*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_21*
validate_shape(*
_output_shapes
:
�
training/Adam/Variable_21/readIdentitytraining/Adam/Variable_21*
T0*,
_class"
 loc:@training/Adam/Variable_21*
_output_shapes
:
p
&training/Adam/zeros_22/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
a
training/Adam/zeros_22/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
training/Adam/zeros_22Fill&training/Adam/zeros_22/shape_as_tensortraining/Adam/zeros_22/Const*
T0*

index_type0*
_output_shapes
:
�
training/Adam/Variable_22
VariableV2*
dtype0*
_output_shapes
:*
	container *
shape:*
shared_name 
�
 training/Adam/Variable_22/AssignAssigntraining/Adam/Variable_22training/Adam/zeros_22*
_output_shapes
:*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_22*
validate_shape(
�
training/Adam/Variable_22/readIdentitytraining/Adam/Variable_22*
_output_shapes
:*
T0*,
_class"
 loc:@training/Adam/Variable_22
p
&training/Adam/zeros_23/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
a
training/Adam/zeros_23/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
training/Adam/zeros_23Fill&training/Adam/zeros_23/shape_as_tensortraining/Adam/zeros_23/Const*
T0*

index_type0*
_output_shapes
:
�
training/Adam/Variable_23
VariableV2*
shape:*
shared_name *
dtype0*
_output_shapes
:*
	container 
�
 training/Adam/Variable_23/AssignAssigntraining/Adam/Variable_23training/Adam/zeros_23*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_23
�
training/Adam/Variable_23/readIdentitytraining/Adam/Variable_23*
T0*,
_class"
 loc:@training/Adam/Variable_23*
_output_shapes
:
r
training/Adam/mul_1MulAdam/beta_1/readtraining/Adam/Variable/read*
T0*
_output_shapes

:+

Z
training/Adam/sub_2/xConst*
_output_shapes
: *
valueB
 *  �?*
dtype0
d
training/Adam/sub_2Subtraining/Adam/sub_2/xAdam/beta_1/read*
T0*
_output_shapes
: 
�
training/Adam/mul_2Multraining/Adam/sub_24training/Adam/gradients/dense_1/MatMul_grad/MatMul_1*
T0*
_output_shapes

:+

m
training/Adam/add_1Addtraining/Adam/mul_1training/Adam/mul_2*
T0*
_output_shapes

:+

t
training/Adam/mul_3MulAdam/beta_2/readtraining/Adam/Variable_8/read*
_output_shapes

:+
*
T0
Z
training/Adam/sub_3/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
d
training/Adam/sub_3Subtraining/Adam/sub_3/xAdam/beta_2/read*
T0*
_output_shapes
: 
}
training/Adam/SquareSquare4training/Adam/gradients/dense_1/MatMul_grad/MatMul_1*
T0*
_output_shapes

:+

n
training/Adam/mul_4Multraining/Adam/sub_3training/Adam/Square*
T0*
_output_shapes

:+

m
training/Adam/add_2Addtraining/Adam/mul_3training/Adam/mul_4*
_output_shapes

:+
*
T0
k
training/Adam/mul_5Multraining/Adam/multraining/Adam/add_1*
_output_shapes

:+
*
T0
Z
training/Adam/Const_2Const*
valueB
 *    *
dtype0*
_output_shapes
: 
Z
training/Adam/Const_3Const*
valueB
 *  �*
dtype0*
_output_shapes
: 
�
%training/Adam/clip_by_value_1/MinimumMinimumtraining/Adam/add_2training/Adam/Const_3*
T0*
_output_shapes

:+

�
training/Adam/clip_by_value_1Maximum%training/Adam/clip_by_value_1/Minimumtraining/Adam/Const_2*
T0*
_output_shapes

:+

d
training/Adam/Sqrt_1Sqrttraining/Adam/clip_by_value_1*
T0*
_output_shapes

:+

Z
training/Adam/add_3/yConst*
valueB
 *���3*
dtype0*
_output_shapes
: 
p
training/Adam/add_3Addtraining/Adam/Sqrt_1training/Adam/add_3/y*
T0*
_output_shapes

:+

u
training/Adam/truediv_1RealDivtraining/Adam/mul_5training/Adam/add_3*
_output_shapes

:+
*
T0
q
training/Adam/sub_4Subdense_1/kernel/readtraining/Adam/truediv_1*
T0*
_output_shapes

:+

�
training/Adam/AssignAssigntraining/Adam/Variabletraining/Adam/add_1*
use_locking(*
T0*)
_class
loc:@training/Adam/Variable*
validate_shape(*
_output_shapes

:+

�
training/Adam/Assign_1Assigntraining/Adam/Variable_8training/Adam/add_2*
validate_shape(*
_output_shapes

:+
*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_8
�
training/Adam/Assign_2Assigndense_1/kerneltraining/Adam/sub_4*
_output_shapes

:+
*
use_locking(*
T0*!
_class
loc:@dense_1/kernel*
validate_shape(
p
training/Adam/mul_6MulAdam/beta_1/readtraining/Adam/Variable_1/read*
T0*
_output_shapes
:

Z
training/Adam/sub_5/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
d
training/Adam/sub_5Subtraining/Adam/sub_5/xAdam/beta_1/read*
_output_shapes
: *
T0
�
training/Adam/mul_7Multraining/Adam/sub_58training/Adam/gradients/dense_1/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:

i
training/Adam/add_4Addtraining/Adam/mul_6training/Adam/mul_7*
T0*
_output_shapes
:

p
training/Adam/mul_8MulAdam/beta_2/readtraining/Adam/Variable_9/read*
_output_shapes
:
*
T0
Z
training/Adam/sub_6/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
d
training/Adam/sub_6Subtraining/Adam/sub_6/xAdam/beta_2/read*
T0*
_output_shapes
: 

training/Adam/Square_1Square8training/Adam/gradients/dense_1/BiasAdd_grad/BiasAddGrad*
_output_shapes
:
*
T0
l
training/Adam/mul_9Multraining/Adam/sub_6training/Adam/Square_1*
T0*
_output_shapes
:

i
training/Adam/add_5Addtraining/Adam/mul_8training/Adam/mul_9*
T0*
_output_shapes
:

h
training/Adam/mul_10Multraining/Adam/multraining/Adam/add_4*
T0*
_output_shapes
:

Z
training/Adam/Const_4Const*
valueB
 *    *
dtype0*
_output_shapes
: 
Z
training/Adam/Const_5Const*
valueB
 *  �*
dtype0*
_output_shapes
: 
�
%training/Adam/clip_by_value_2/MinimumMinimumtraining/Adam/add_5training/Adam/Const_5*
_output_shapes
:
*
T0
�
training/Adam/clip_by_value_2Maximum%training/Adam/clip_by_value_2/Minimumtraining/Adam/Const_4*
T0*
_output_shapes
:

`
training/Adam/Sqrt_2Sqrttraining/Adam/clip_by_value_2*
T0*
_output_shapes
:

Z
training/Adam/add_6/yConst*
valueB
 *���3*
dtype0*
_output_shapes
: 
l
training/Adam/add_6Addtraining/Adam/Sqrt_2training/Adam/add_6/y*
T0*
_output_shapes
:

r
training/Adam/truediv_2RealDivtraining/Adam/mul_10training/Adam/add_6*
_output_shapes
:
*
T0
k
training/Adam/sub_7Subdense_1/bias/readtraining/Adam/truediv_2*
T0*
_output_shapes
:

�
training/Adam/Assign_3Assigntraining/Adam/Variable_1training/Adam/add_4*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_1*
validate_shape(*
_output_shapes
:

�
training/Adam/Assign_4Assigntraining/Adam/Variable_9training/Adam/add_5*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_9*
validate_shape(*
_output_shapes
:

�
training/Adam/Assign_5Assigndense_1/biastraining/Adam/sub_7*
use_locking(*
T0*
_class
loc:@dense_1/bias*
validate_shape(*
_output_shapes
:

u
training/Adam/mul_11MulAdam/beta_1/readtraining/Adam/Variable_2/read*
T0*
_output_shapes

:

Z
training/Adam/sub_8/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
d
training/Adam/sub_8Subtraining/Adam/sub_8/xAdam/beta_1/read*
T0*
_output_shapes
: 
�
training/Adam/mul_12Multraining/Adam/sub_84training/Adam/gradients/dense_2/MatMul_grad/MatMul_1*
T0*
_output_shapes

:

o
training/Adam/add_7Addtraining/Adam/mul_11training/Adam/mul_12*
_output_shapes

:
*
T0
v
training/Adam/mul_13MulAdam/beta_2/readtraining/Adam/Variable_10/read*
T0*
_output_shapes

:

Z
training/Adam/sub_9/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
d
training/Adam/sub_9Subtraining/Adam/sub_9/xAdam/beta_2/read*
T0*
_output_shapes
: 

training/Adam/Square_2Square4training/Adam/gradients/dense_2/MatMul_grad/MatMul_1*
T0*
_output_shapes

:

q
training/Adam/mul_14Multraining/Adam/sub_9training/Adam/Square_2*
T0*
_output_shapes

:

o
training/Adam/add_8Addtraining/Adam/mul_13training/Adam/mul_14*
T0*
_output_shapes

:

l
training/Adam/mul_15Multraining/Adam/multraining/Adam/add_7*
_output_shapes

:
*
T0
Z
training/Adam/Const_6Const*
dtype0*
_output_shapes
: *
valueB
 *    
Z
training/Adam/Const_7Const*
valueB
 *  �*
dtype0*
_output_shapes
: 
�
%training/Adam/clip_by_value_3/MinimumMinimumtraining/Adam/add_8training/Adam/Const_7*
T0*
_output_shapes

:

�
training/Adam/clip_by_value_3Maximum%training/Adam/clip_by_value_3/Minimumtraining/Adam/Const_6*
T0*
_output_shapes

:

d
training/Adam/Sqrt_3Sqrttraining/Adam/clip_by_value_3*
T0*
_output_shapes

:

Z
training/Adam/add_9/yConst*
dtype0*
_output_shapes
: *
valueB
 *���3
p
training/Adam/add_9Addtraining/Adam/Sqrt_3training/Adam/add_9/y*
_output_shapes

:
*
T0
v
training/Adam/truediv_3RealDivtraining/Adam/mul_15training/Adam/add_9*
_output_shapes

:
*
T0
r
training/Adam/sub_10Subdense_2/kernel/readtraining/Adam/truediv_3*
T0*
_output_shapes

:

�
training/Adam/Assign_6Assigntraining/Adam/Variable_2training/Adam/add_7*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_2*
validate_shape(*
_output_shapes

:

�
training/Adam/Assign_7Assigntraining/Adam/Variable_10training/Adam/add_8*
_output_shapes

:
*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_10*
validate_shape(
�
training/Adam/Assign_8Assigndense_2/kerneltraining/Adam/sub_10*
_output_shapes

:
*
use_locking(*
T0*!
_class
loc:@dense_2/kernel*
validate_shape(
q
training/Adam/mul_16MulAdam/beta_1/readtraining/Adam/Variable_3/read*
_output_shapes
:*
T0
[
training/Adam/sub_11/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_11Subtraining/Adam/sub_11/xAdam/beta_1/read*
_output_shapes
: *
T0
�
training/Adam/mul_17Multraining/Adam/sub_118training/Adam/gradients/dense_2/BiasAdd_grad/BiasAddGrad*
_output_shapes
:*
T0
l
training/Adam/add_10Addtraining/Adam/mul_16training/Adam/mul_17*
T0*
_output_shapes
:
r
training/Adam/mul_18MulAdam/beta_2/readtraining/Adam/Variable_11/read*
T0*
_output_shapes
:
[
training/Adam/sub_12/xConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
f
training/Adam/sub_12Subtraining/Adam/sub_12/xAdam/beta_2/read*
T0*
_output_shapes
: 

training/Adam/Square_3Square8training/Adam/gradients/dense_2/BiasAdd_grad/BiasAddGrad*
_output_shapes
:*
T0
n
training/Adam/mul_19Multraining/Adam/sub_12training/Adam/Square_3*
T0*
_output_shapes
:
l
training/Adam/add_11Addtraining/Adam/mul_18training/Adam/mul_19*
T0*
_output_shapes
:
i
training/Adam/mul_20Multraining/Adam/multraining/Adam/add_10*
_output_shapes
:*
T0
Z
training/Adam/Const_8Const*
valueB
 *    *
dtype0*
_output_shapes
: 
Z
training/Adam/Const_9Const*
valueB
 *  �*
dtype0*
_output_shapes
: 
�
%training/Adam/clip_by_value_4/MinimumMinimumtraining/Adam/add_11training/Adam/Const_9*
_output_shapes
:*
T0
�
training/Adam/clip_by_value_4Maximum%training/Adam/clip_by_value_4/Minimumtraining/Adam/Const_8*
T0*
_output_shapes
:
`
training/Adam/Sqrt_4Sqrttraining/Adam/clip_by_value_4*
T0*
_output_shapes
:
[
training/Adam/add_12/yConst*
dtype0*
_output_shapes
: *
valueB
 *���3
n
training/Adam/add_12Addtraining/Adam/Sqrt_4training/Adam/add_12/y*
T0*
_output_shapes
:
s
training/Adam/truediv_4RealDivtraining/Adam/mul_20training/Adam/add_12*
T0*
_output_shapes
:
l
training/Adam/sub_13Subdense_2/bias/readtraining/Adam/truediv_4*
T0*
_output_shapes
:
�
training/Adam/Assign_9Assigntraining/Adam/Variable_3training/Adam/add_10*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_3*
validate_shape(*
_output_shapes
:
�
training/Adam/Assign_10Assigntraining/Adam/Variable_11training/Adam/add_11*,
_class"
 loc:@training/Adam/Variable_11*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
�
training/Adam/Assign_11Assigndense_2/biastraining/Adam/sub_13*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*
_class
loc:@dense_2/bias
u
training/Adam/mul_21MulAdam/beta_1/readtraining/Adam/Variable_4/read*
_output_shapes

:*
T0
[
training/Adam/sub_14/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_14Subtraining/Adam/sub_14/xAdam/beta_1/read*
_output_shapes
: *
T0
�
training/Adam/mul_22Multraining/Adam/sub_144training/Adam/gradients/dense_3/MatMul_grad/MatMul_1*
_output_shapes

:*
T0
p
training/Adam/add_13Addtraining/Adam/mul_21training/Adam/mul_22*
_output_shapes

:*
T0
v
training/Adam/mul_23MulAdam/beta_2/readtraining/Adam/Variable_12/read*
T0*
_output_shapes

:
[
training/Adam/sub_15/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_15Subtraining/Adam/sub_15/xAdam/beta_2/read*
T0*
_output_shapes
: 

training/Adam/Square_4Square4training/Adam/gradients/dense_3/MatMul_grad/MatMul_1*
_output_shapes

:*
T0
r
training/Adam/mul_24Multraining/Adam/sub_15training/Adam/Square_4*
T0*
_output_shapes

:
p
training/Adam/add_14Addtraining/Adam/mul_23training/Adam/mul_24*
T0*
_output_shapes

:
m
training/Adam/mul_25Multraining/Adam/multraining/Adam/add_13*
T0*
_output_shapes

:
[
training/Adam/Const_10Const*
valueB
 *    *
dtype0*
_output_shapes
: 
[
training/Adam/Const_11Const*
valueB
 *  �*
dtype0*
_output_shapes
: 
�
%training/Adam/clip_by_value_5/MinimumMinimumtraining/Adam/add_14training/Adam/Const_11*
T0*
_output_shapes

:
�
training/Adam/clip_by_value_5Maximum%training/Adam/clip_by_value_5/Minimumtraining/Adam/Const_10*
_output_shapes

:*
T0
d
training/Adam/Sqrt_5Sqrttraining/Adam/clip_by_value_5*
T0*
_output_shapes

:
[
training/Adam/add_15/yConst*
dtype0*
_output_shapes
: *
valueB
 *���3
r
training/Adam/add_15Addtraining/Adam/Sqrt_5training/Adam/add_15/y*
T0*
_output_shapes

:
w
training/Adam/truediv_5RealDivtraining/Adam/mul_25training/Adam/add_15*
T0*
_output_shapes

:
r
training/Adam/sub_16Subdense_3/kernel/readtraining/Adam/truediv_5*
T0*
_output_shapes

:
�
training/Adam/Assign_12Assigntraining/Adam/Variable_4training/Adam/add_13*
T0*+
_class!
loc:@training/Adam/Variable_4*
validate_shape(*
_output_shapes

:*
use_locking(
�
training/Adam/Assign_13Assigntraining/Adam/Variable_12training/Adam/add_14*
_output_shapes

:*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_12*
validate_shape(
�
training/Adam/Assign_14Assigndense_3/kerneltraining/Adam/sub_16*
use_locking(*
T0*!
_class
loc:@dense_3/kernel*
validate_shape(*
_output_shapes

:
q
training/Adam/mul_26MulAdam/beta_1/readtraining/Adam/Variable_5/read*
T0*
_output_shapes
:
[
training/Adam/sub_17/xConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
f
training/Adam/sub_17Subtraining/Adam/sub_17/xAdam/beta_1/read*
T0*
_output_shapes
: 
�
training/Adam/mul_27Multraining/Adam/sub_178training/Adam/gradients/dense_3/BiasAdd_grad/BiasAddGrad*
_output_shapes
:*
T0
l
training/Adam/add_16Addtraining/Adam/mul_26training/Adam/mul_27*
T0*
_output_shapes
:
r
training/Adam/mul_28MulAdam/beta_2/readtraining/Adam/Variable_13/read*
_output_shapes
:*
T0
[
training/Adam/sub_18/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_18Subtraining/Adam/sub_18/xAdam/beta_2/read*
T0*
_output_shapes
: 

training/Adam/Square_5Square8training/Adam/gradients/dense_3/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:
n
training/Adam/mul_29Multraining/Adam/sub_18training/Adam/Square_5*
T0*
_output_shapes
:
l
training/Adam/add_17Addtraining/Adam/mul_28training/Adam/mul_29*
T0*
_output_shapes
:
i
training/Adam/mul_30Multraining/Adam/multraining/Adam/add_16*
T0*
_output_shapes
:
[
training/Adam/Const_12Const*
valueB
 *    *
dtype0*
_output_shapes
: 
[
training/Adam/Const_13Const*
valueB
 *  �*
dtype0*
_output_shapes
: 
�
%training/Adam/clip_by_value_6/MinimumMinimumtraining/Adam/add_17training/Adam/Const_13*
T0*
_output_shapes
:
�
training/Adam/clip_by_value_6Maximum%training/Adam/clip_by_value_6/Minimumtraining/Adam/Const_12*
T0*
_output_shapes
:
`
training/Adam/Sqrt_6Sqrttraining/Adam/clip_by_value_6*
T0*
_output_shapes
:
[
training/Adam/add_18/yConst*
valueB
 *���3*
dtype0*
_output_shapes
: 
n
training/Adam/add_18Addtraining/Adam/Sqrt_6training/Adam/add_18/y*
T0*
_output_shapes
:
s
training/Adam/truediv_6RealDivtraining/Adam/mul_30training/Adam/add_18*
T0*
_output_shapes
:
l
training/Adam/sub_19Subdense_3/bias/readtraining/Adam/truediv_6*
T0*
_output_shapes
:
�
training/Adam/Assign_15Assigntraining/Adam/Variable_5training/Adam/add_16*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_5*
validate_shape(*
_output_shapes
:
�
training/Adam/Assign_16Assigntraining/Adam/Variable_13training/Adam/add_17*,
_class"
 loc:@training/Adam/Variable_13*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
�
training/Adam/Assign_17Assigndense_3/biastraining/Adam/sub_19*
use_locking(*
T0*
_class
loc:@dense_3/bias*
validate_shape(*
_output_shapes
:
u
training/Adam/mul_31MulAdam/beta_1/readtraining/Adam/Variable_6/read*
T0*
_output_shapes

:+
[
training/Adam/sub_20/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_20Subtraining/Adam/sub_20/xAdam/beta_1/read*
T0*
_output_shapes
: 
�
training/Adam/mul_32Multraining/Adam/sub_204training/Adam/gradients/dense_4/MatMul_grad/MatMul_1*
T0*
_output_shapes

:+
p
training/Adam/add_19Addtraining/Adam/mul_31training/Adam/mul_32*
_output_shapes

:+*
T0
v
training/Adam/mul_33MulAdam/beta_2/readtraining/Adam/Variable_14/read*
T0*
_output_shapes

:+
[
training/Adam/sub_21/xConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
f
training/Adam/sub_21Subtraining/Adam/sub_21/xAdam/beta_2/read*
T0*
_output_shapes
: 

training/Adam/Square_6Square4training/Adam/gradients/dense_4/MatMul_grad/MatMul_1*
_output_shapes

:+*
T0
r
training/Adam/mul_34Multraining/Adam/sub_21training/Adam/Square_6*
T0*
_output_shapes

:+
p
training/Adam/add_20Addtraining/Adam/mul_33training/Adam/mul_34*
T0*
_output_shapes

:+
m
training/Adam/mul_35Multraining/Adam/multraining/Adam/add_19*
T0*
_output_shapes

:+
[
training/Adam/Const_14Const*
valueB
 *    *
dtype0*
_output_shapes
: 
[
training/Adam/Const_15Const*
valueB
 *  �*
dtype0*
_output_shapes
: 
�
%training/Adam/clip_by_value_7/MinimumMinimumtraining/Adam/add_20training/Adam/Const_15*
_output_shapes

:+*
T0
�
training/Adam/clip_by_value_7Maximum%training/Adam/clip_by_value_7/Minimumtraining/Adam/Const_14*
T0*
_output_shapes

:+
d
training/Adam/Sqrt_7Sqrttraining/Adam/clip_by_value_7*
T0*
_output_shapes

:+
[
training/Adam/add_21/yConst*
valueB
 *���3*
dtype0*
_output_shapes
: 
r
training/Adam/add_21Addtraining/Adam/Sqrt_7training/Adam/add_21/y*
_output_shapes

:+*
T0
w
training/Adam/truediv_7RealDivtraining/Adam/mul_35training/Adam/add_21*
T0*
_output_shapes

:+
r
training/Adam/sub_22Subdense_4/kernel/readtraining/Adam/truediv_7*
_output_shapes

:+*
T0
�
training/Adam/Assign_18Assigntraining/Adam/Variable_6training/Adam/add_19*
_output_shapes

:+*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_6*
validate_shape(
�
training/Adam/Assign_19Assigntraining/Adam/Variable_14training/Adam/add_20*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_14*
validate_shape(*
_output_shapes

:+
�
training/Adam/Assign_20Assigndense_4/kerneltraining/Adam/sub_22*
use_locking(*
T0*!
_class
loc:@dense_4/kernel*
validate_shape(*
_output_shapes

:+
q
training/Adam/mul_36MulAdam/beta_1/readtraining/Adam/Variable_7/read*
_output_shapes
:+*
T0
[
training/Adam/sub_23/xConst*
_output_shapes
: *
valueB
 *  �?*
dtype0
f
training/Adam/sub_23Subtraining/Adam/sub_23/xAdam/beta_1/read*
T0*
_output_shapes
: 
�
training/Adam/mul_37Multraining/Adam/sub_238training/Adam/gradients/dense_4/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:+
l
training/Adam/add_22Addtraining/Adam/mul_36training/Adam/mul_37*
_output_shapes
:+*
T0
r
training/Adam/mul_38MulAdam/beta_2/readtraining/Adam/Variable_15/read*
T0*
_output_shapes
:+
[
training/Adam/sub_24/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_24Subtraining/Adam/sub_24/xAdam/beta_2/read*
T0*
_output_shapes
: 

training/Adam/Square_7Square8training/Adam/gradients/dense_4/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:+
n
training/Adam/mul_39Multraining/Adam/sub_24training/Adam/Square_7*
T0*
_output_shapes
:+
l
training/Adam/add_23Addtraining/Adam/mul_38training/Adam/mul_39*
T0*
_output_shapes
:+
i
training/Adam/mul_40Multraining/Adam/multraining/Adam/add_22*
T0*
_output_shapes
:+
[
training/Adam/Const_16Const*
_output_shapes
: *
valueB
 *    *
dtype0
[
training/Adam/Const_17Const*
valueB
 *  �*
dtype0*
_output_shapes
: 
�
%training/Adam/clip_by_value_8/MinimumMinimumtraining/Adam/add_23training/Adam/Const_17*
_output_shapes
:+*
T0
�
training/Adam/clip_by_value_8Maximum%training/Adam/clip_by_value_8/Minimumtraining/Adam/Const_16*
T0*
_output_shapes
:+
`
training/Adam/Sqrt_8Sqrttraining/Adam/clip_by_value_8*
T0*
_output_shapes
:+
[
training/Adam/add_24/yConst*
valueB
 *���3*
dtype0*
_output_shapes
: 
n
training/Adam/add_24Addtraining/Adam/Sqrt_8training/Adam/add_24/y*
_output_shapes
:+*
T0
s
training/Adam/truediv_8RealDivtraining/Adam/mul_40training/Adam/add_24*
_output_shapes
:+*
T0
l
training/Adam/sub_25Subdense_4/bias/readtraining/Adam/truediv_8*
_output_shapes
:+*
T0
�
training/Adam/Assign_21Assigntraining/Adam/Variable_7training/Adam/add_22*
validate_shape(*
_output_shapes
:+*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_7
�
training/Adam/Assign_22Assigntraining/Adam/Variable_15training/Adam/add_23*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_15*
validate_shape(*
_output_shapes
:+
�
training/Adam/Assign_23Assigndense_4/biastraining/Adam/sub_25*
_output_shapes
:+*
use_locking(*
T0*
_class
loc:@dense_4/bias*
validate_shape(
�
training/group_depsNoOp	^loss/add^metrics/acc/Mean^training/Adam/Assign^training/Adam/AssignAdd^training/Adam/Assign_1^training/Adam/Assign_10^training/Adam/Assign_11^training/Adam/Assign_12^training/Adam/Assign_13^training/Adam/Assign_14^training/Adam/Assign_15^training/Adam/Assign_16^training/Adam/Assign_17^training/Adam/Assign_18^training/Adam/Assign_19^training/Adam/Assign_2^training/Adam/Assign_20^training/Adam/Assign_21^training/Adam/Assign_22^training/Adam/Assign_23^training/Adam/Assign_3^training/Adam/Assign_4^training/Adam/Assign_5^training/Adam/Assign_6^training/Adam/Assign_7^training/Adam/Assign_8^training/Adam/Assign_9
0

group_depsNoOp	^loss/add^metrics/acc/Mean
�
IsVariableInitializedIsVariableInitializeddense_1/kernel*!
_class
loc:@dense_1/kernel*
dtype0*
_output_shapes
: 
�
IsVariableInitialized_1IsVariableInitializeddense_1/bias*
_class
loc:@dense_1/bias*
dtype0*
_output_shapes
: 
�
IsVariableInitialized_2IsVariableInitializeddense_2/kernel*!
_class
loc:@dense_2/kernel*
dtype0*
_output_shapes
: 
�
IsVariableInitialized_3IsVariableInitializeddense_2/bias*
_class
loc:@dense_2/bias*
dtype0*
_output_shapes
: 
�
IsVariableInitialized_4IsVariableInitializeddense_3/kernel*
dtype0*
_output_shapes
: *!
_class
loc:@dense_3/kernel
�
IsVariableInitialized_5IsVariableInitializeddense_3/bias*
_class
loc:@dense_3/bias*
dtype0*
_output_shapes
: 
�
IsVariableInitialized_6IsVariableInitializeddense_4/kernel*!
_class
loc:@dense_4/kernel*
dtype0*
_output_shapes
: 
�
IsVariableInitialized_7IsVariableInitializeddense_4/bias*
_class
loc:@dense_4/bias*
dtype0*
_output_shapes
: 
�
IsVariableInitialized_8IsVariableInitializedAdam/iterations*
_output_shapes
: *"
_class
loc:@Adam/iterations*
dtype0	
z
IsVariableInitialized_9IsVariableInitializedAdam/lr*
_class
loc:@Adam/lr*
dtype0*
_output_shapes
: 
�
IsVariableInitialized_10IsVariableInitializedAdam/beta_1*
_class
loc:@Adam/beta_1*
dtype0*
_output_shapes
: 
�
IsVariableInitialized_11IsVariableInitializedAdam/beta_2*
_class
loc:@Adam/beta_2*
dtype0*
_output_shapes
: 
�
IsVariableInitialized_12IsVariableInitialized
Adam/decay*
_class
loc:@Adam/decay*
dtype0*
_output_shapes
: 
�
IsVariableInitialized_13IsVariableInitializedtraining/Adam/Variable*
dtype0*
_output_shapes
: *)
_class
loc:@training/Adam/Variable
�
IsVariableInitialized_14IsVariableInitializedtraining/Adam/Variable_1*+
_class!
loc:@training/Adam/Variable_1*
dtype0*
_output_shapes
: 
�
IsVariableInitialized_15IsVariableInitializedtraining/Adam/Variable_2*+
_class!
loc:@training/Adam/Variable_2*
dtype0*
_output_shapes
: 
�
IsVariableInitialized_16IsVariableInitializedtraining/Adam/Variable_3*+
_class!
loc:@training/Adam/Variable_3*
dtype0*
_output_shapes
: 
�
IsVariableInitialized_17IsVariableInitializedtraining/Adam/Variable_4*+
_class!
loc:@training/Adam/Variable_4*
dtype0*
_output_shapes
: 
�
IsVariableInitialized_18IsVariableInitializedtraining/Adam/Variable_5*+
_class!
loc:@training/Adam/Variable_5*
dtype0*
_output_shapes
: 
�
IsVariableInitialized_19IsVariableInitializedtraining/Adam/Variable_6*
_output_shapes
: *+
_class!
loc:@training/Adam/Variable_6*
dtype0
�
IsVariableInitialized_20IsVariableInitializedtraining/Adam/Variable_7*+
_class!
loc:@training/Adam/Variable_7*
dtype0*
_output_shapes
: 
�
IsVariableInitialized_21IsVariableInitializedtraining/Adam/Variable_8*+
_class!
loc:@training/Adam/Variable_8*
dtype0*
_output_shapes
: 
�
IsVariableInitialized_22IsVariableInitializedtraining/Adam/Variable_9*
dtype0*
_output_shapes
: *+
_class!
loc:@training/Adam/Variable_9
�
IsVariableInitialized_23IsVariableInitializedtraining/Adam/Variable_10*,
_class"
 loc:@training/Adam/Variable_10*
dtype0*
_output_shapes
: 
�
IsVariableInitialized_24IsVariableInitializedtraining/Adam/Variable_11*,
_class"
 loc:@training/Adam/Variable_11*
dtype0*
_output_shapes
: 
�
IsVariableInitialized_25IsVariableInitializedtraining/Adam/Variable_12*
_output_shapes
: *,
_class"
 loc:@training/Adam/Variable_12*
dtype0
�
IsVariableInitialized_26IsVariableInitializedtraining/Adam/Variable_13*
dtype0*
_output_shapes
: *,
_class"
 loc:@training/Adam/Variable_13
�
IsVariableInitialized_27IsVariableInitializedtraining/Adam/Variable_14*,
_class"
 loc:@training/Adam/Variable_14*
dtype0*
_output_shapes
: 
�
IsVariableInitialized_28IsVariableInitializedtraining/Adam/Variable_15*,
_class"
 loc:@training/Adam/Variable_15*
dtype0*
_output_shapes
: 
�
IsVariableInitialized_29IsVariableInitializedtraining/Adam/Variable_16*
dtype0*
_output_shapes
: *,
_class"
 loc:@training/Adam/Variable_16
�
IsVariableInitialized_30IsVariableInitializedtraining/Adam/Variable_17*,
_class"
 loc:@training/Adam/Variable_17*
dtype0*
_output_shapes
: 
�
IsVariableInitialized_31IsVariableInitializedtraining/Adam/Variable_18*,
_class"
 loc:@training/Adam/Variable_18*
dtype0*
_output_shapes
: 
�
IsVariableInitialized_32IsVariableInitializedtraining/Adam/Variable_19*,
_class"
 loc:@training/Adam/Variable_19*
dtype0*
_output_shapes
: 
�
IsVariableInitialized_33IsVariableInitializedtraining/Adam/Variable_20*,
_class"
 loc:@training/Adam/Variable_20*
dtype0*
_output_shapes
: 
�
IsVariableInitialized_34IsVariableInitializedtraining/Adam/Variable_21*,
_class"
 loc:@training/Adam/Variable_21*
dtype0*
_output_shapes
: 
�
IsVariableInitialized_35IsVariableInitializedtraining/Adam/Variable_22*,
_class"
 loc:@training/Adam/Variable_22*
dtype0*
_output_shapes
: 
�
IsVariableInitialized_36IsVariableInitializedtraining/Adam/Variable_23*,
_class"
 loc:@training/Adam/Variable_23*
dtype0*
_output_shapes
: 
�
initNoOp^Adam/beta_1/Assign^Adam/beta_2/Assign^Adam/decay/Assign^Adam/iterations/Assign^Adam/lr/Assign^dense_1/bias/Assign^dense_1/kernel/Assign^dense_2/bias/Assign^dense_2/kernel/Assign^dense_3/bias/Assign^dense_3/kernel/Assign^dense_4/bias/Assign^dense_4/kernel/Assign^training/Adam/Variable/Assign ^training/Adam/Variable_1/Assign!^training/Adam/Variable_10/Assign!^training/Adam/Variable_11/Assign!^training/Adam/Variable_12/Assign!^training/Adam/Variable_13/Assign!^training/Adam/Variable_14/Assign!^training/Adam/Variable_15/Assign!^training/Adam/Variable_16/Assign!^training/Adam/Variable_17/Assign!^training/Adam/Variable_18/Assign!^training/Adam/Variable_19/Assign ^training/Adam/Variable_2/Assign!^training/Adam/Variable_20/Assign!^training/Adam/Variable_21/Assign!^training/Adam/Variable_22/Assign!^training/Adam/Variable_23/Assign ^training/Adam/Variable_3/Assign ^training/Adam/Variable_4/Assign ^training/Adam/Variable_5/Assign ^training/Adam/Variable_6/Assign ^training/Adam/Variable_7/Assign ^training/Adam/Variable_8/Assign ^training/Adam/Variable_9/Assign"�E*�     �:S[	��B�B=�AJ��
��
,
Abs
x"T
y"T"
Ttype:

2	
:
Add
x"T
y"T
z"T"
Ttype:
2	
W
AddN
inputs"T*N
sum"T"
Nint(0"!
Ttype:
2	��
�
ArgMax

input"T
	dimension"Tidx
output"output_type" 
Ttype:
2	"
Tidxtype0:
2	"
output_typetype0	:
2	
x
Assign
ref"T�

value"T

output_ref"T�"	
Ttype"
validate_shapebool("
use_lockingbool(�
s
	AssignAdd
ref"T�

value"T

output_ref"T�" 
Ttype:
2	"
use_lockingbool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
~
BiasAddGrad
out_backprop"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
R
BroadcastGradientArgs
s0"T
s1"T
r0"T
r1"T"
Ttype0:
2	
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
8
Const
output"dtype"
valuetensor"
dtypetype
S
DynamicStitch
indices*N
data"T*N
merged"T"
Nint(0"	
Ttype
B
Equal
x"T
y"T
z
"
Ttype:
2	
�
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
?
FloorDiv
x"T
y"T
z"T"
Ttype:
2	
9
FloorMod
x"T
y"T
z"T"
Ttype:

2	
.
Identity

input"T
output"T"	
Ttype
N
IsVariableInitialized
ref"dtype�
is_initialized
"
dtypetype�
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
;
Maximum
x"T
y"T
z"T"
Ttype:

2	�
�
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
;
Minimum
x"T
y"T
z"T"
Ttype:

2	�
=
Mul
x"T
y"T
z"T"
Ttype:
2	�
.
Neg
x"T
y"T"
Ttype:

2	

NoOp
E
NotEqual
x"T
y"T
z
"
Ttype:
2	
�
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
6
Pow
x"T
y"T
z"T"
Ttype:

2	
�
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
~
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	�
a
Range
start"Tidx
limit"Tidx
delta"Tidx
output"Tidx"
Tidxtype0:	
2	
>
RealDiv
x"T
y"T
z"T"
Ttype:
2	
E
Relu
features"T
activations"T"
Ttype:
2	
V
ReluGrad
	gradients"T
features"T
	backprops"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
/
Sign
x"T
y"T"
Ttype:

2	
-
Sqrt
x"T
y"T"
Ttype:

2
1
Square
x"T
y"T"
Ttype:

2	
:
Sub
x"T
y"T
z"T"
Ttype:
2	
�
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
-
Tanh
x"T
y"T"
Ttype:

2
:
TanhGrad
y"T
dy"T
z"T"
Ttype:

2
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	
s

VariableV2
ref"dtype�"
shapeshape"
dtypetype"
	containerstring "
shared_namestring �*1.13.12
b'unknown'��
j
input_1Placeholder*
shape:���������+*
dtype0*'
_output_shapes
:���������+
m
dense_1/random_uniform/shapeConst*
valueB"+   
   *
dtype0*
_output_shapes
:
_
dense_1/random_uniform/minConst*
valueB
 *�D��*
dtype0*
_output_shapes
: 
_
dense_1/random_uniform/maxConst*
valueB
 *�D�>*
dtype0*
_output_shapes
: 
�
$dense_1/random_uniform/RandomUniformRandomUniformdense_1/random_uniform/shape*
seed2끞*
_output_shapes

:+
*

seed*
T0*
dtype0
z
dense_1/random_uniform/subSubdense_1/random_uniform/maxdense_1/random_uniform/min*
T0*
_output_shapes
: 
�
dense_1/random_uniform/mulMul$dense_1/random_uniform/RandomUniformdense_1/random_uniform/sub*
_output_shapes

:+
*
T0
~
dense_1/random_uniformAdddense_1/random_uniform/muldense_1/random_uniform/min*
T0*
_output_shapes

:+

�
dense_1/kernel
VariableV2*
shape
:+
*
shared_name *
dtype0*
	container *
_output_shapes

:+

�
dense_1/kernel/AssignAssigndense_1/kerneldense_1/random_uniform*
validate_shape(*
_output_shapes

:+
*
use_locking(*
T0*!
_class
loc:@dense_1/kernel
{
dense_1/kernel/readIdentitydense_1/kernel*
T0*!
_class
loc:@dense_1/kernel*
_output_shapes

:+

Z
dense_1/ConstConst*
_output_shapes
:
*
valueB
*    *
dtype0
x
dense_1/bias
VariableV2*
shared_name *
dtype0*
	container *
_output_shapes
:
*
shape:

�
dense_1/bias/AssignAssigndense_1/biasdense_1/Const*
_class
loc:@dense_1/bias*
validate_shape(*
_output_shapes
:
*
use_locking(*
T0
q
dense_1/bias/readIdentitydense_1/bias*
_output_shapes
:
*
T0*
_class
loc:@dense_1/bias
�
dense_1/MatMulMatMulinput_1dense_1/kernel/read*
T0*
transpose_a( *'
_output_shapes
:���������
*
transpose_b( 
�
dense_1/BiasAddBiasAdddense_1/MatMuldense_1/bias/read*'
_output_shapes
:���������
*
T0*
data_formatNHWC
W
dense_1/TanhTanhdense_1/BiasAdd*
T0*'
_output_shapes
:���������

g
 dense_1/activity_regularizer/AbsAbsdense_1/Tanh*
T0*'
_output_shapes
:���������

g
"dense_1/activity_regularizer/mul/xConst*
valueB
 *o�:*
dtype0*
_output_shapes
: 
�
 dense_1/activity_regularizer/mulMul"dense_1/activity_regularizer/mul/x dense_1/activity_regularizer/Abs*
T0*'
_output_shapes
:���������

s
"dense_1/activity_regularizer/ConstConst*
valueB"       *
dtype0*
_output_shapes
:
�
 dense_1/activity_regularizer/SumSum dense_1/activity_regularizer/mul"dense_1/activity_regularizer/Const*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
g
"dense_1/activity_regularizer/add/xConst*
dtype0*
_output_shapes
: *
valueB
 *    
�
 dense_1/activity_regularizer/addAdd"dense_1/activity_regularizer/add/x dense_1/activity_regularizer/Sum*
_output_shapes
: *
T0
m
dense_2/random_uniform/shapeConst*
valueB"
      *
dtype0*
_output_shapes
:
_
dense_2/random_uniform/minConst*
valueB
 *��!�*
dtype0*
_output_shapes
: 
_
dense_2/random_uniform/maxConst*
_output_shapes
: *
valueB
 *��!?*
dtype0
�
$dense_2/random_uniform/RandomUniformRandomUniformdense_2/random_uniform/shape*
dtype0*
seed2�ߜ*
_output_shapes

:
*

seed*
T0
z
dense_2/random_uniform/subSubdense_2/random_uniform/maxdense_2/random_uniform/min*
T0*
_output_shapes
: 
�
dense_2/random_uniform/mulMul$dense_2/random_uniform/RandomUniformdense_2/random_uniform/sub*
T0*
_output_shapes

:

~
dense_2/random_uniformAdddense_2/random_uniform/muldense_2/random_uniform/min*
T0*
_output_shapes

:

�
dense_2/kernel
VariableV2*
shared_name *
dtype0*
	container *
_output_shapes

:
*
shape
:

�
dense_2/kernel/AssignAssigndense_2/kerneldense_2/random_uniform*
validate_shape(*
_output_shapes

:
*
use_locking(*
T0*!
_class
loc:@dense_2/kernel
{
dense_2/kernel/readIdentitydense_2/kernel*
T0*!
_class
loc:@dense_2/kernel*
_output_shapes

:

Z
dense_2/ConstConst*
valueB*    *
dtype0*
_output_shapes
:
x
dense_2/bias
VariableV2*
	container *
_output_shapes
:*
shape:*
shared_name *
dtype0
�
dense_2/bias/AssignAssigndense_2/biasdense_2/Const*
use_locking(*
T0*
_class
loc:@dense_2/bias*
validate_shape(*
_output_shapes
:
q
dense_2/bias/readIdentitydense_2/bias*
_output_shapes
:*
T0*
_class
loc:@dense_2/bias
�
dense_2/MatMulMatMuldense_1/Tanhdense_2/kernel/read*
T0*
transpose_a( *'
_output_shapes
:���������*
transpose_b( 
�
dense_2/BiasAddBiasAdddense_2/MatMuldense_2/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:���������
W
dense_2/ReluReludense_2/BiasAdd*'
_output_shapes
:���������*
T0
m
dense_3/random_uniform/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
_
dense_3/random_uniform/minConst*
_output_shapes
: *
valueB
 *�KF�*
dtype0
_
dense_3/random_uniform/maxConst*
_output_shapes
: *
valueB
 *�KF?*
dtype0
�
$dense_3/random_uniform/RandomUniformRandomUniformdense_3/random_uniform/shape*
dtype0*
seed2��*
_output_shapes

:*

seed*
T0
z
dense_3/random_uniform/subSubdense_3/random_uniform/maxdense_3/random_uniform/min*
T0*
_output_shapes
: 
�
dense_3/random_uniform/mulMul$dense_3/random_uniform/RandomUniformdense_3/random_uniform/sub*
T0*
_output_shapes

:
~
dense_3/random_uniformAdddense_3/random_uniform/muldense_3/random_uniform/min*
T0*
_output_shapes

:
�
dense_3/kernel
VariableV2*
dtype0*
	container *
_output_shapes

:*
shape
:*
shared_name 
�
dense_3/kernel/AssignAssigndense_3/kerneldense_3/random_uniform*
_output_shapes

:*
use_locking(*
T0*!
_class
loc:@dense_3/kernel*
validate_shape(
{
dense_3/kernel/readIdentitydense_3/kernel*
T0*!
_class
loc:@dense_3/kernel*
_output_shapes

:
Z
dense_3/ConstConst*
valueB*    *
dtype0*
_output_shapes
:
x
dense_3/bias
VariableV2*
shared_name *
dtype0*
	container *
_output_shapes
:*
shape:
�
dense_3/bias/AssignAssigndense_3/biasdense_3/Const*
_class
loc:@dense_3/bias*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
q
dense_3/bias/readIdentitydense_3/bias*
_class
loc:@dense_3/bias*
_output_shapes
:*
T0
�
dense_3/MatMulMatMuldense_2/Reludense_3/kernel/read*
T0*
transpose_a( *'
_output_shapes
:���������*
transpose_b( 
�
dense_3/BiasAddBiasAdddense_3/MatMuldense_3/bias/read*
data_formatNHWC*'
_output_shapes
:���������*
T0
W
dense_3/TanhTanhdense_3/BiasAdd*
T0*'
_output_shapes
:���������
m
dense_4/random_uniform/shapeConst*
valueB"   +   *
dtype0*
_output_shapes
:
_
dense_4/random_uniform/minConst*
valueB
 *���*
dtype0*
_output_shapes
: 
_
dense_4/random_uniform/maxConst*
valueB
 *��>*
dtype0*
_output_shapes
: 
�
$dense_4/random_uniform/RandomUniformRandomUniformdense_4/random_uniform/shape*
dtype0*
seed2�Ֆ*
_output_shapes

:+*

seed*
T0
z
dense_4/random_uniform/subSubdense_4/random_uniform/maxdense_4/random_uniform/min*
_output_shapes
: *
T0
�
dense_4/random_uniform/mulMul$dense_4/random_uniform/RandomUniformdense_4/random_uniform/sub*
_output_shapes

:+*
T0
~
dense_4/random_uniformAdddense_4/random_uniform/muldense_4/random_uniform/min*
T0*
_output_shapes

:+
�
dense_4/kernel
VariableV2*
shared_name *
dtype0*
	container *
_output_shapes

:+*
shape
:+
�
dense_4/kernel/AssignAssigndense_4/kerneldense_4/random_uniform*
use_locking(*
T0*!
_class
loc:@dense_4/kernel*
validate_shape(*
_output_shapes

:+
{
dense_4/kernel/readIdentitydense_4/kernel*!
_class
loc:@dense_4/kernel*
_output_shapes

:+*
T0
Z
dense_4/ConstConst*
valueB+*    *
dtype0*
_output_shapes
:+
x
dense_4/bias
VariableV2*
shape:+*
shared_name *
dtype0*
	container *
_output_shapes
:+
�
dense_4/bias/AssignAssigndense_4/biasdense_4/Const*
use_locking(*
T0*
_class
loc:@dense_4/bias*
validate_shape(*
_output_shapes
:+
q
dense_4/bias/readIdentitydense_4/bias*
_output_shapes
:+*
T0*
_class
loc:@dense_4/bias
�
dense_4/MatMulMatMuldense_3/Tanhdense_4/kernel/read*
transpose_a( *'
_output_shapes
:���������+*
transpose_b( *
T0
�
dense_4/BiasAddBiasAdddense_4/MatMuldense_4/bias/read*
data_formatNHWC*'
_output_shapes
:���������+*
T0
W
dense_4/ReluReludense_4/BiasAdd*
T0*'
_output_shapes
:���������+
_
Adam/iterations/initial_valueConst*
value	B	 R *
dtype0	*
_output_shapes
: 
s
Adam/iterations
VariableV2*
shape: *
shared_name *
dtype0	*
	container *
_output_shapes
: 
�
Adam/iterations/AssignAssignAdam/iterationsAdam/iterations/initial_value*
validate_shape(*
_output_shapes
: *
use_locking(*
T0	*"
_class
loc:@Adam/iterations
v
Adam/iterations/readIdentityAdam/iterations*
_output_shapes
: *
T0	*"
_class
loc:@Adam/iterations
Z
Adam/lr/initial_valueConst*
valueB
 *o�:*
dtype0*
_output_shapes
: 
k
Adam/lr
VariableV2*
shared_name *
dtype0*
	container *
_output_shapes
: *
shape: 
�
Adam/lr/AssignAssignAdam/lrAdam/lr/initial_value*
use_locking(*
T0*
_class
loc:@Adam/lr*
validate_shape(*
_output_shapes
: 
^
Adam/lr/readIdentityAdam/lr*
T0*
_class
loc:@Adam/lr*
_output_shapes
: 
^
Adam/beta_1/initial_valueConst*
dtype0*
_output_shapes
: *
valueB
 *fff?
o
Adam/beta_1
VariableV2*
	container *
_output_shapes
: *
shape: *
shared_name *
dtype0
�
Adam/beta_1/AssignAssignAdam/beta_1Adam/beta_1/initial_value*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@Adam/beta_1
j
Adam/beta_1/readIdentityAdam/beta_1*
T0*
_class
loc:@Adam/beta_1*
_output_shapes
: 
^
Adam/beta_2/initial_valueConst*
valueB
 *w�?*
dtype0*
_output_shapes
: 
o
Adam/beta_2
VariableV2*
shared_name *
dtype0*
	container *
_output_shapes
: *
shape: 
�
Adam/beta_2/AssignAssignAdam/beta_2Adam/beta_2/initial_value*
T0*
_class
loc:@Adam/beta_2*
validate_shape(*
_output_shapes
: *
use_locking(
j
Adam/beta_2/readIdentityAdam/beta_2*
_output_shapes
: *
T0*
_class
loc:@Adam/beta_2
]
Adam/decay/initial_valueConst*
valueB
 *    *
dtype0*
_output_shapes
: 
n

Adam/decay
VariableV2*
dtype0*
	container *
_output_shapes
: *
shape: *
shared_name 
�
Adam/decay/AssignAssign
Adam/decayAdam/decay/initial_value*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@Adam/decay*
validate_shape(
g
Adam/decay/readIdentity
Adam/decay*
T0*
_class
loc:@Adam/decay*
_output_shapes
: 
�
dense_4_targetPlaceholder*
dtype0*0
_output_shapes
:������������������*%
shape:������������������
q
dense_4_sample_weightsPlaceholder*
dtype0*#
_output_shapes
:���������*
shape:���������
l
loss/dense_4_loss/subSubdense_4/Reludense_4_target*'
_output_shapes
:���������+*
T0
k
loss/dense_4_loss/SquareSquareloss/dense_4_loss/sub*
T0*'
_output_shapes
:���������+
s
(loss/dense_4_loss/Mean/reduction_indicesConst*
valueB :
���������*
dtype0*
_output_shapes
: 
�
loss/dense_4_loss/MeanMeanloss/dense_4_loss/Square(loss/dense_4_loss/Mean/reduction_indices*
T0*#
_output_shapes
:���������*

Tidx0*
	keep_dims( 
m
*loss/dense_4_loss/Mean_1/reduction_indicesConst*
valueB *
dtype0*
_output_shapes
: 
�
loss/dense_4_loss/Mean_1Meanloss/dense_4_loss/Mean*loss/dense_4_loss/Mean_1/reduction_indices*
T0*#
_output_shapes
:���������*

Tidx0*
	keep_dims( 
|
loss/dense_4_loss/mulMulloss/dense_4_loss/Mean_1dense_4_sample_weights*
T0*#
_output_shapes
:���������
a
loss/dense_4_loss/NotEqual/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
loss/dense_4_loss/NotEqualNotEqualdense_4_sample_weightsloss/dense_4_loss/NotEqual/y*
T0*#
_output_shapes
:���������
�
loss/dense_4_loss/CastCastloss/dense_4_loss/NotEqual*

DstT0*#
_output_shapes
:���������*

SrcT0
*
Truncate( 
a
loss/dense_4_loss/ConstConst*
valueB: *
dtype0*
_output_shapes
:
�
loss/dense_4_loss/Mean_2Meanloss/dense_4_loss/Castloss/dense_4_loss/Const*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
�
loss/dense_4_loss/truedivRealDivloss/dense_4_loss/mulloss/dense_4_loss/Mean_2*
T0*#
_output_shapes
:���������
c
loss/dense_4_loss/Const_1Const*
_output_shapes
:*
valueB: *
dtype0
�
loss/dense_4_loss/Mean_3Meanloss/dense_4_loss/truedivloss/dense_4_loss/Const_1*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
O

loss/mul/xConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
V
loss/mulMul
loss/mul/xloss/dense_4_loss/Mean_3*
T0*
_output_shapes
: 
\
loss/addAddloss/mul dense_1/activity_regularizer/add*
T0*
_output_shapes
: 
g
metrics/acc/ArgMax/dimensionConst*
valueB :
���������*
dtype0*
_output_shapes
: 
�
metrics/acc/ArgMaxArgMaxdense_4_targetmetrics/acc/ArgMax/dimension*
T0*
output_type0	*#
_output_shapes
:���������*

Tidx0
i
metrics/acc/ArgMax_1/dimensionConst*
_output_shapes
: *
valueB :
���������*
dtype0
�
metrics/acc/ArgMax_1ArgMaxdense_4/Relumetrics/acc/ArgMax_1/dimension*
T0*
output_type0	*#
_output_shapes
:���������*

Tidx0
r
metrics/acc/EqualEqualmetrics/acc/ArgMaxmetrics/acc/ArgMax_1*
T0	*#
_output_shapes
:���������
x
metrics/acc/CastCastmetrics/acc/Equal*
Truncate( *

DstT0*#
_output_shapes
:���������*

SrcT0

[
metrics/acc/ConstConst*
valueB: *
dtype0*
_output_shapes
:
{
metrics/acc/MeanMeanmetrics/acc/Castmetrics/acc/Const*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
}
training/Adam/gradients/ShapeConst*
_output_shapes
: *
_class
loc:@loss/add*
valueB *
dtype0
�
!training/Adam/gradients/grad_ys_0Const*
_class
loc:@loss/add*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
training/Adam/gradients/FillFilltraining/Adam/gradients/Shape!training/Adam/gradients/grad_ys_0*
T0*
_class
loc:@loss/add*

index_type0*
_output_shapes
: 
�
)training/Adam/gradients/loss/mul_grad/MulMultraining/Adam/gradients/Fillloss/dense_4_loss/Mean_3*
T0*
_class
loc:@loss/mul*
_output_shapes
: 
�
+training/Adam/gradients/loss/mul_grad/Mul_1Multraining/Adam/gradients/Fill
loss/mul/x*
T0*
_class
loc:@loss/mul*
_output_shapes
: 
�
Ctraining/Adam/gradients/loss/dense_4_loss/Mean_3_grad/Reshape/shapeConst*
dtype0*
_output_shapes
:*+
_class!
loc:@loss/dense_4_loss/Mean_3*
valueB:
�
=training/Adam/gradients/loss/dense_4_loss/Mean_3_grad/ReshapeReshape+training/Adam/gradients/loss/mul_grad/Mul_1Ctraining/Adam/gradients/loss/dense_4_loss/Mean_3_grad/Reshape/shape*
T0*+
_class!
loc:@loss/dense_4_loss/Mean_3*
Tshape0*
_output_shapes
:
�
;training/Adam/gradients/loss/dense_4_loss/Mean_3_grad/ShapeShapeloss/dense_4_loss/truediv*
T0*+
_class!
loc:@loss/dense_4_loss/Mean_3*
out_type0*
_output_shapes
:
�
:training/Adam/gradients/loss/dense_4_loss/Mean_3_grad/TileTile=training/Adam/gradients/loss/dense_4_loss/Mean_3_grad/Reshape;training/Adam/gradients/loss/dense_4_loss/Mean_3_grad/Shape*

Tmultiples0*
T0*+
_class!
loc:@loss/dense_4_loss/Mean_3*#
_output_shapes
:���������
�
=training/Adam/gradients/loss/dense_4_loss/Mean_3_grad/Shape_1Shapeloss/dense_4_loss/truediv*
_output_shapes
:*
T0*+
_class!
loc:@loss/dense_4_loss/Mean_3*
out_type0
�
=training/Adam/gradients/loss/dense_4_loss/Mean_3_grad/Shape_2Const*
_output_shapes
: *+
_class!
loc:@loss/dense_4_loss/Mean_3*
valueB *
dtype0
�
;training/Adam/gradients/loss/dense_4_loss/Mean_3_grad/ConstConst*
_output_shapes
:*+
_class!
loc:@loss/dense_4_loss/Mean_3*
valueB: *
dtype0
�
:training/Adam/gradients/loss/dense_4_loss/Mean_3_grad/ProdProd=training/Adam/gradients/loss/dense_4_loss/Mean_3_grad/Shape_1;training/Adam/gradients/loss/dense_4_loss/Mean_3_grad/Const*
T0*+
_class!
loc:@loss/dense_4_loss/Mean_3*
_output_shapes
: *
	keep_dims( *

Tidx0
�
=training/Adam/gradients/loss/dense_4_loss/Mean_3_grad/Const_1Const*
dtype0*
_output_shapes
:*+
_class!
loc:@loss/dense_4_loss/Mean_3*
valueB: 
�
<training/Adam/gradients/loss/dense_4_loss/Mean_3_grad/Prod_1Prod=training/Adam/gradients/loss/dense_4_loss/Mean_3_grad/Shape_2=training/Adam/gradients/loss/dense_4_loss/Mean_3_grad/Const_1*
T0*+
_class!
loc:@loss/dense_4_loss/Mean_3*
_output_shapes
: *
	keep_dims( *

Tidx0
�
?training/Adam/gradients/loss/dense_4_loss/Mean_3_grad/Maximum/yConst*+
_class!
loc:@loss/dense_4_loss/Mean_3*
value	B :*
dtype0*
_output_shapes
: 
�
=training/Adam/gradients/loss/dense_4_loss/Mean_3_grad/MaximumMaximum<training/Adam/gradients/loss/dense_4_loss/Mean_3_grad/Prod_1?training/Adam/gradients/loss/dense_4_loss/Mean_3_grad/Maximum/y*
T0*+
_class!
loc:@loss/dense_4_loss/Mean_3*
_output_shapes
: 
�
>training/Adam/gradients/loss/dense_4_loss/Mean_3_grad/floordivFloorDiv:training/Adam/gradients/loss/dense_4_loss/Mean_3_grad/Prod=training/Adam/gradients/loss/dense_4_loss/Mean_3_grad/Maximum*
T0*+
_class!
loc:@loss/dense_4_loss/Mean_3*
_output_shapes
: 
�
:training/Adam/gradients/loss/dense_4_loss/Mean_3_grad/CastCast>training/Adam/gradients/loss/dense_4_loss/Mean_3_grad/floordiv*

SrcT0*+
_class!
loc:@loss/dense_4_loss/Mean_3*
Truncate( *

DstT0*
_output_shapes
: 
�
=training/Adam/gradients/loss/dense_4_loss/Mean_3_grad/truedivRealDiv:training/Adam/gradients/loss/dense_4_loss/Mean_3_grad/Tile:training/Adam/gradients/loss/dense_4_loss/Mean_3_grad/Cast*
T0*+
_class!
loc:@loss/dense_4_loss/Mean_3*#
_output_shapes
:���������
�
Ktraining/Adam/gradients/dense_1/activity_regularizer/Sum_grad/Reshape/shapeConst*3
_class)
'%loc:@dense_1/activity_regularizer/Sum*
valueB"      *
dtype0*
_output_shapes
:
�
Etraining/Adam/gradients/dense_1/activity_regularizer/Sum_grad/ReshapeReshapetraining/Adam/gradients/FillKtraining/Adam/gradients/dense_1/activity_regularizer/Sum_grad/Reshape/shape*
T0*3
_class)
'%loc:@dense_1/activity_regularizer/Sum*
Tshape0*
_output_shapes

:
�
Ctraining/Adam/gradients/dense_1/activity_regularizer/Sum_grad/ShapeShape dense_1/activity_regularizer/mul*
T0*3
_class)
'%loc:@dense_1/activity_regularizer/Sum*
out_type0*
_output_shapes
:
�
Btraining/Adam/gradients/dense_1/activity_regularizer/Sum_grad/TileTileEtraining/Adam/gradients/dense_1/activity_regularizer/Sum_grad/ReshapeCtraining/Adam/gradients/dense_1/activity_regularizer/Sum_grad/Shape*'
_output_shapes
:���������
*

Tmultiples0*
T0*3
_class)
'%loc:@dense_1/activity_regularizer/Sum
�
<training/Adam/gradients/loss/dense_4_loss/truediv_grad/ShapeShapeloss/dense_4_loss/mul*
_output_shapes
:*
T0*,
_class"
 loc:@loss/dense_4_loss/truediv*
out_type0
�
>training/Adam/gradients/loss/dense_4_loss/truediv_grad/Shape_1Const*,
_class"
 loc:@loss/dense_4_loss/truediv*
valueB *
dtype0*
_output_shapes
: 
�
Ltraining/Adam/gradients/loss/dense_4_loss/truediv_grad/BroadcastGradientArgsBroadcastGradientArgs<training/Adam/gradients/loss/dense_4_loss/truediv_grad/Shape>training/Adam/gradients/loss/dense_4_loss/truediv_grad/Shape_1*
T0*,
_class"
 loc:@loss/dense_4_loss/truediv*2
_output_shapes 
:���������:���������
�
>training/Adam/gradients/loss/dense_4_loss/truediv_grad/RealDivRealDiv=training/Adam/gradients/loss/dense_4_loss/Mean_3_grad/truedivloss/dense_4_loss/Mean_2*
T0*,
_class"
 loc:@loss/dense_4_loss/truediv*#
_output_shapes
:���������
�
:training/Adam/gradients/loss/dense_4_loss/truediv_grad/SumSum>training/Adam/gradients/loss/dense_4_loss/truediv_grad/RealDivLtraining/Adam/gradients/loss/dense_4_loss/truediv_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*,
_class"
 loc:@loss/dense_4_loss/truediv*
_output_shapes
:
�
>training/Adam/gradients/loss/dense_4_loss/truediv_grad/ReshapeReshape:training/Adam/gradients/loss/dense_4_loss/truediv_grad/Sum<training/Adam/gradients/loss/dense_4_loss/truediv_grad/Shape*
T0*,
_class"
 loc:@loss/dense_4_loss/truediv*
Tshape0*#
_output_shapes
:���������
�
:training/Adam/gradients/loss/dense_4_loss/truediv_grad/NegNegloss/dense_4_loss/mul*
T0*,
_class"
 loc:@loss/dense_4_loss/truediv*#
_output_shapes
:���������
�
@training/Adam/gradients/loss/dense_4_loss/truediv_grad/RealDiv_1RealDiv:training/Adam/gradients/loss/dense_4_loss/truediv_grad/Negloss/dense_4_loss/Mean_2*,
_class"
 loc:@loss/dense_4_loss/truediv*#
_output_shapes
:���������*
T0
�
@training/Adam/gradients/loss/dense_4_loss/truediv_grad/RealDiv_2RealDiv@training/Adam/gradients/loss/dense_4_loss/truediv_grad/RealDiv_1loss/dense_4_loss/Mean_2*#
_output_shapes
:���������*
T0*,
_class"
 loc:@loss/dense_4_loss/truediv
�
:training/Adam/gradients/loss/dense_4_loss/truediv_grad/mulMul=training/Adam/gradients/loss/dense_4_loss/Mean_3_grad/truediv@training/Adam/gradients/loss/dense_4_loss/truediv_grad/RealDiv_2*
T0*,
_class"
 loc:@loss/dense_4_loss/truediv*#
_output_shapes
:���������
�
<training/Adam/gradients/loss/dense_4_loss/truediv_grad/Sum_1Sum:training/Adam/gradients/loss/dense_4_loss/truediv_grad/mulNtraining/Adam/gradients/loss/dense_4_loss/truediv_grad/BroadcastGradientArgs:1*,
_class"
 loc:@loss/dense_4_loss/truediv*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
@training/Adam/gradients/loss/dense_4_loss/truediv_grad/Reshape_1Reshape<training/Adam/gradients/loss/dense_4_loss/truediv_grad/Sum_1>training/Adam/gradients/loss/dense_4_loss/truediv_grad/Shape_1*,
_class"
 loc:@loss/dense_4_loss/truediv*
Tshape0*
_output_shapes
: *
T0
�
Ctraining/Adam/gradients/dense_1/activity_regularizer/mul_grad/ShapeConst*3
_class)
'%loc:@dense_1/activity_regularizer/mul*
valueB *
dtype0*
_output_shapes
: 
�
Etraining/Adam/gradients/dense_1/activity_regularizer/mul_grad/Shape_1Shape dense_1/activity_regularizer/Abs*
T0*3
_class)
'%loc:@dense_1/activity_regularizer/mul*
out_type0*
_output_shapes
:
�
Straining/Adam/gradients/dense_1/activity_regularizer/mul_grad/BroadcastGradientArgsBroadcastGradientArgsCtraining/Adam/gradients/dense_1/activity_regularizer/mul_grad/ShapeEtraining/Adam/gradients/dense_1/activity_regularizer/mul_grad/Shape_1*
T0*3
_class)
'%loc:@dense_1/activity_regularizer/mul*2
_output_shapes 
:���������:���������
�
Atraining/Adam/gradients/dense_1/activity_regularizer/mul_grad/MulMulBtraining/Adam/gradients/dense_1/activity_regularizer/Sum_grad/Tile dense_1/activity_regularizer/Abs*
T0*3
_class)
'%loc:@dense_1/activity_regularizer/mul*'
_output_shapes
:���������

�
Atraining/Adam/gradients/dense_1/activity_regularizer/mul_grad/SumSumAtraining/Adam/gradients/dense_1/activity_regularizer/mul_grad/MulStraining/Adam/gradients/dense_1/activity_regularizer/mul_grad/BroadcastGradientArgs*
T0*3
_class)
'%loc:@dense_1/activity_regularizer/mul*
_output_shapes
:*
	keep_dims( *

Tidx0
�
Etraining/Adam/gradients/dense_1/activity_regularizer/mul_grad/ReshapeReshapeAtraining/Adam/gradients/dense_1/activity_regularizer/mul_grad/SumCtraining/Adam/gradients/dense_1/activity_regularizer/mul_grad/Shape*
T0*3
_class)
'%loc:@dense_1/activity_regularizer/mul*
Tshape0*
_output_shapes
: 
�
Ctraining/Adam/gradients/dense_1/activity_regularizer/mul_grad/Mul_1Mul"dense_1/activity_regularizer/mul/xBtraining/Adam/gradients/dense_1/activity_regularizer/Sum_grad/Tile*'
_output_shapes
:���������
*
T0*3
_class)
'%loc:@dense_1/activity_regularizer/mul
�
Ctraining/Adam/gradients/dense_1/activity_regularizer/mul_grad/Sum_1SumCtraining/Adam/gradients/dense_1/activity_regularizer/mul_grad/Mul_1Utraining/Adam/gradients/dense_1/activity_regularizer/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0*3
_class)
'%loc:@dense_1/activity_regularizer/mul
�
Gtraining/Adam/gradients/dense_1/activity_regularizer/mul_grad/Reshape_1ReshapeCtraining/Adam/gradients/dense_1/activity_regularizer/mul_grad/Sum_1Etraining/Adam/gradients/dense_1/activity_regularizer/mul_grad/Shape_1*'
_output_shapes
:���������
*
T0*3
_class)
'%loc:@dense_1/activity_regularizer/mul*
Tshape0
�
8training/Adam/gradients/loss/dense_4_loss/mul_grad/ShapeShapeloss/dense_4_loss/Mean_1*
T0*(
_class
loc:@loss/dense_4_loss/mul*
out_type0*
_output_shapes
:
�
:training/Adam/gradients/loss/dense_4_loss/mul_grad/Shape_1Shapedense_4_sample_weights*
_output_shapes
:*
T0*(
_class
loc:@loss/dense_4_loss/mul*
out_type0
�
Htraining/Adam/gradients/loss/dense_4_loss/mul_grad/BroadcastGradientArgsBroadcastGradientArgs8training/Adam/gradients/loss/dense_4_loss/mul_grad/Shape:training/Adam/gradients/loss/dense_4_loss/mul_grad/Shape_1*
T0*(
_class
loc:@loss/dense_4_loss/mul*2
_output_shapes 
:���������:���������
�
6training/Adam/gradients/loss/dense_4_loss/mul_grad/MulMul>training/Adam/gradients/loss/dense_4_loss/truediv_grad/Reshapedense_4_sample_weights*
T0*(
_class
loc:@loss/dense_4_loss/mul*#
_output_shapes
:���������
�
6training/Adam/gradients/loss/dense_4_loss/mul_grad/SumSum6training/Adam/gradients/loss/dense_4_loss/mul_grad/MulHtraining/Adam/gradients/loss/dense_4_loss/mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*(
_class
loc:@loss/dense_4_loss/mul*
_output_shapes
:
�
:training/Adam/gradients/loss/dense_4_loss/mul_grad/ReshapeReshape6training/Adam/gradients/loss/dense_4_loss/mul_grad/Sum8training/Adam/gradients/loss/dense_4_loss/mul_grad/Shape*(
_class
loc:@loss/dense_4_loss/mul*
Tshape0*#
_output_shapes
:���������*
T0
�
8training/Adam/gradients/loss/dense_4_loss/mul_grad/Mul_1Mulloss/dense_4_loss/Mean_1>training/Adam/gradients/loss/dense_4_loss/truediv_grad/Reshape*
T0*(
_class
loc:@loss/dense_4_loss/mul*#
_output_shapes
:���������
�
8training/Adam/gradients/loss/dense_4_loss/mul_grad/Sum_1Sum8training/Adam/gradients/loss/dense_4_loss/mul_grad/Mul_1Jtraining/Adam/gradients/loss/dense_4_loss/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*(
_class
loc:@loss/dense_4_loss/mul*
_output_shapes
:
�
<training/Adam/gradients/loss/dense_4_loss/mul_grad/Reshape_1Reshape8training/Adam/gradients/loss/dense_4_loss/mul_grad/Sum_1:training/Adam/gradients/loss/dense_4_loss/mul_grad/Shape_1*
T0*(
_class
loc:@loss/dense_4_loss/mul*
Tshape0*#
_output_shapes
:���������
�
Btraining/Adam/gradients/dense_1/activity_regularizer/Abs_grad/SignSigndense_1/Tanh*
T0*3
_class)
'%loc:@dense_1/activity_regularizer/Abs*'
_output_shapes
:���������

�
Atraining/Adam/gradients/dense_1/activity_regularizer/Abs_grad/mulMulGtraining/Adam/gradients/dense_1/activity_regularizer/mul_grad/Reshape_1Btraining/Adam/gradients/dense_1/activity_regularizer/Abs_grad/Sign*
T0*3
_class)
'%loc:@dense_1/activity_regularizer/Abs*'
_output_shapes
:���������

�
;training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/ShapeShapeloss/dense_4_loss/Mean*
T0*+
_class!
loc:@loss/dense_4_loss/Mean_1*
out_type0*
_output_shapes
:
�
:training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/SizeConst*+
_class!
loc:@loss/dense_4_loss/Mean_1*
value	B :*
dtype0*
_output_shapes
: 
�
9training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/addAdd*loss/dense_4_loss/Mean_1/reduction_indices:training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Size*
T0*+
_class!
loc:@loss/dense_4_loss/Mean_1*
_output_shapes
: 
�
9training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/modFloorMod9training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/add:training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Size*
T0*+
_class!
loc:@loss/dense_4_loss/Mean_1*
_output_shapes
: 
�
=training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Shape_1Const*+
_class!
loc:@loss/dense_4_loss/Mean_1*
valueB: *
dtype0*
_output_shapes
:
�
Atraining/Adam/gradients/loss/dense_4_loss/Mean_1_grad/range/startConst*+
_class!
loc:@loss/dense_4_loss/Mean_1*
value	B : *
dtype0*
_output_shapes
: 
�
Atraining/Adam/gradients/loss/dense_4_loss/Mean_1_grad/range/deltaConst*+
_class!
loc:@loss/dense_4_loss/Mean_1*
value	B :*
dtype0*
_output_shapes
: 
�
;training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/rangeRangeAtraining/Adam/gradients/loss/dense_4_loss/Mean_1_grad/range/start:training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/SizeAtraining/Adam/gradients/loss/dense_4_loss/Mean_1_grad/range/delta*
_output_shapes
:*

Tidx0*+
_class!
loc:@loss/dense_4_loss/Mean_1
�
@training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Fill/valueConst*+
_class!
loc:@loss/dense_4_loss/Mean_1*
value	B :*
dtype0*
_output_shapes
: 
�
:training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/FillFill=training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Shape_1@training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Fill/value*
T0*+
_class!
loc:@loss/dense_4_loss/Mean_1*

index_type0*
_output_shapes
: 
�
Ctraining/Adam/gradients/loss/dense_4_loss/Mean_1_grad/DynamicStitchDynamicStitch;training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/range9training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/mod;training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Shape:training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Fill*
T0*+
_class!
loc:@loss/dense_4_loss/Mean_1*
N*
_output_shapes
:
�
?training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Maximum/yConst*+
_class!
loc:@loss/dense_4_loss/Mean_1*
value	B :*
dtype0*
_output_shapes
: 
�
=training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/MaximumMaximumCtraining/Adam/gradients/loss/dense_4_loss/Mean_1_grad/DynamicStitch?training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Maximum/y*
_output_shapes
:*
T0*+
_class!
loc:@loss/dense_4_loss/Mean_1
�
>training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/floordivFloorDiv;training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Shape=training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Maximum*
_output_shapes
:*
T0*+
_class!
loc:@loss/dense_4_loss/Mean_1
�
=training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/ReshapeReshape:training/Adam/gradients/loss/dense_4_loss/mul_grad/ReshapeCtraining/Adam/gradients/loss/dense_4_loss/Mean_1_grad/DynamicStitch*
T0*+
_class!
loc:@loss/dense_4_loss/Mean_1*
Tshape0*#
_output_shapes
:���������
�
:training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/TileTile=training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Reshape>training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/floordiv*

Tmultiples0*
T0*+
_class!
loc:@loss/dense_4_loss/Mean_1*#
_output_shapes
:���������
�
=training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Shape_2Shapeloss/dense_4_loss/Mean*
T0*+
_class!
loc:@loss/dense_4_loss/Mean_1*
out_type0*
_output_shapes
:
�
=training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Shape_3Shapeloss/dense_4_loss/Mean_1*
_output_shapes
:*
T0*+
_class!
loc:@loss/dense_4_loss/Mean_1*
out_type0
�
;training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/ConstConst*+
_class!
loc:@loss/dense_4_loss/Mean_1*
valueB: *
dtype0*
_output_shapes
:
�
:training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/ProdProd=training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Shape_2;training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Const*
	keep_dims( *

Tidx0*
T0*+
_class!
loc:@loss/dense_4_loss/Mean_1*
_output_shapes
: 
�
=training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Const_1Const*+
_class!
loc:@loss/dense_4_loss/Mean_1*
valueB: *
dtype0*
_output_shapes
:
�
<training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Prod_1Prod=training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Shape_3=training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Const_1*
T0*+
_class!
loc:@loss/dense_4_loss/Mean_1*
_output_shapes
: *
	keep_dims( *

Tidx0
�
Atraining/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Maximum_1/yConst*+
_class!
loc:@loss/dense_4_loss/Mean_1*
value	B :*
dtype0*
_output_shapes
: 
�
?training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Maximum_1Maximum<training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Prod_1Atraining/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Maximum_1/y*
T0*+
_class!
loc:@loss/dense_4_loss/Mean_1*
_output_shapes
: 
�
@training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/floordiv_1FloorDiv:training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Prod?training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Maximum_1*
_output_shapes
: *
T0*+
_class!
loc:@loss/dense_4_loss/Mean_1
�
:training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/CastCast@training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/floordiv_1*

DstT0*
_output_shapes
: *

SrcT0*+
_class!
loc:@loss/dense_4_loss/Mean_1*
Truncate( 
�
=training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/truedivRealDiv:training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Tile:training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Cast*+
_class!
loc:@loss/dense_4_loss/Mean_1*#
_output_shapes
:���������*
T0
�
9training/Adam/gradients/loss/dense_4_loss/Mean_grad/ShapeShapeloss/dense_4_loss/Square*)
_class
loc:@loss/dense_4_loss/Mean*
out_type0*
_output_shapes
:*
T0
�
8training/Adam/gradients/loss/dense_4_loss/Mean_grad/SizeConst*)
_class
loc:@loss/dense_4_loss/Mean*
value	B :*
dtype0*
_output_shapes
: 
�
7training/Adam/gradients/loss/dense_4_loss/Mean_grad/addAdd(loss/dense_4_loss/Mean/reduction_indices8training/Adam/gradients/loss/dense_4_loss/Mean_grad/Size*
T0*)
_class
loc:@loss/dense_4_loss/Mean*
_output_shapes
: 
�
7training/Adam/gradients/loss/dense_4_loss/Mean_grad/modFloorMod7training/Adam/gradients/loss/dense_4_loss/Mean_grad/add8training/Adam/gradients/loss/dense_4_loss/Mean_grad/Size*
_output_shapes
: *
T0*)
_class
loc:@loss/dense_4_loss/Mean
�
;training/Adam/gradients/loss/dense_4_loss/Mean_grad/Shape_1Const*)
_class
loc:@loss/dense_4_loss/Mean*
valueB *
dtype0*
_output_shapes
: 
�
?training/Adam/gradients/loss/dense_4_loss/Mean_grad/range/startConst*
dtype0*
_output_shapes
: *)
_class
loc:@loss/dense_4_loss/Mean*
value	B : 
�
?training/Adam/gradients/loss/dense_4_loss/Mean_grad/range/deltaConst*)
_class
loc:@loss/dense_4_loss/Mean*
value	B :*
dtype0*
_output_shapes
: 
�
9training/Adam/gradients/loss/dense_4_loss/Mean_grad/rangeRange?training/Adam/gradients/loss/dense_4_loss/Mean_grad/range/start8training/Adam/gradients/loss/dense_4_loss/Mean_grad/Size?training/Adam/gradients/loss/dense_4_loss/Mean_grad/range/delta*
_output_shapes
:*

Tidx0*)
_class
loc:@loss/dense_4_loss/Mean
�
>training/Adam/gradients/loss/dense_4_loss/Mean_grad/Fill/valueConst*
dtype0*
_output_shapes
: *)
_class
loc:@loss/dense_4_loss/Mean*
value	B :
�
8training/Adam/gradients/loss/dense_4_loss/Mean_grad/FillFill;training/Adam/gradients/loss/dense_4_loss/Mean_grad/Shape_1>training/Adam/gradients/loss/dense_4_loss/Mean_grad/Fill/value*
_output_shapes
: *
T0*)
_class
loc:@loss/dense_4_loss/Mean*

index_type0
�
Atraining/Adam/gradients/loss/dense_4_loss/Mean_grad/DynamicStitchDynamicStitch9training/Adam/gradients/loss/dense_4_loss/Mean_grad/range7training/Adam/gradients/loss/dense_4_loss/Mean_grad/mod9training/Adam/gradients/loss/dense_4_loss/Mean_grad/Shape8training/Adam/gradients/loss/dense_4_loss/Mean_grad/Fill*
T0*)
_class
loc:@loss/dense_4_loss/Mean*
N*
_output_shapes
:
�
=training/Adam/gradients/loss/dense_4_loss/Mean_grad/Maximum/yConst*
_output_shapes
: *)
_class
loc:@loss/dense_4_loss/Mean*
value	B :*
dtype0
�
;training/Adam/gradients/loss/dense_4_loss/Mean_grad/MaximumMaximumAtraining/Adam/gradients/loss/dense_4_loss/Mean_grad/DynamicStitch=training/Adam/gradients/loss/dense_4_loss/Mean_grad/Maximum/y*
T0*)
_class
loc:@loss/dense_4_loss/Mean*
_output_shapes
:
�
<training/Adam/gradients/loss/dense_4_loss/Mean_grad/floordivFloorDiv9training/Adam/gradients/loss/dense_4_loss/Mean_grad/Shape;training/Adam/gradients/loss/dense_4_loss/Mean_grad/Maximum*
T0*)
_class
loc:@loss/dense_4_loss/Mean*
_output_shapes
:
�
;training/Adam/gradients/loss/dense_4_loss/Mean_grad/ReshapeReshape=training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/truedivAtraining/Adam/gradients/loss/dense_4_loss/Mean_grad/DynamicStitch*
T0*)
_class
loc:@loss/dense_4_loss/Mean*
Tshape0*0
_output_shapes
:������������������
�
8training/Adam/gradients/loss/dense_4_loss/Mean_grad/TileTile;training/Adam/gradients/loss/dense_4_loss/Mean_grad/Reshape<training/Adam/gradients/loss/dense_4_loss/Mean_grad/floordiv*0
_output_shapes
:������������������*

Tmultiples0*
T0*)
_class
loc:@loss/dense_4_loss/Mean
�
;training/Adam/gradients/loss/dense_4_loss/Mean_grad/Shape_2Shapeloss/dense_4_loss/Square*
T0*)
_class
loc:@loss/dense_4_loss/Mean*
out_type0*
_output_shapes
:
�
;training/Adam/gradients/loss/dense_4_loss/Mean_grad/Shape_3Shapeloss/dense_4_loss/Mean*
T0*)
_class
loc:@loss/dense_4_loss/Mean*
out_type0*
_output_shapes
:
�
9training/Adam/gradients/loss/dense_4_loss/Mean_grad/ConstConst*)
_class
loc:@loss/dense_4_loss/Mean*
valueB: *
dtype0*
_output_shapes
:
�
8training/Adam/gradients/loss/dense_4_loss/Mean_grad/ProdProd;training/Adam/gradients/loss/dense_4_loss/Mean_grad/Shape_29training/Adam/gradients/loss/dense_4_loss/Mean_grad/Const*
	keep_dims( *

Tidx0*
T0*)
_class
loc:@loss/dense_4_loss/Mean*
_output_shapes
: 
�
;training/Adam/gradients/loss/dense_4_loss/Mean_grad/Const_1Const*
_output_shapes
:*)
_class
loc:@loss/dense_4_loss/Mean*
valueB: *
dtype0
�
:training/Adam/gradients/loss/dense_4_loss/Mean_grad/Prod_1Prod;training/Adam/gradients/loss/dense_4_loss/Mean_grad/Shape_3;training/Adam/gradients/loss/dense_4_loss/Mean_grad/Const_1*)
_class
loc:@loss/dense_4_loss/Mean*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
�
?training/Adam/gradients/loss/dense_4_loss/Mean_grad/Maximum_1/yConst*)
_class
loc:@loss/dense_4_loss/Mean*
value	B :*
dtype0*
_output_shapes
: 
�
=training/Adam/gradients/loss/dense_4_loss/Mean_grad/Maximum_1Maximum:training/Adam/gradients/loss/dense_4_loss/Mean_grad/Prod_1?training/Adam/gradients/loss/dense_4_loss/Mean_grad/Maximum_1/y*
T0*)
_class
loc:@loss/dense_4_loss/Mean*
_output_shapes
: 
�
>training/Adam/gradients/loss/dense_4_loss/Mean_grad/floordiv_1FloorDiv8training/Adam/gradients/loss/dense_4_loss/Mean_grad/Prod=training/Adam/gradients/loss/dense_4_loss/Mean_grad/Maximum_1*
_output_shapes
: *
T0*)
_class
loc:@loss/dense_4_loss/Mean
�
8training/Adam/gradients/loss/dense_4_loss/Mean_grad/CastCast>training/Adam/gradients/loss/dense_4_loss/Mean_grad/floordiv_1*

SrcT0*)
_class
loc:@loss/dense_4_loss/Mean*
Truncate( *

DstT0*
_output_shapes
: 
�
;training/Adam/gradients/loss/dense_4_loss/Mean_grad/truedivRealDiv8training/Adam/gradients/loss/dense_4_loss/Mean_grad/Tile8training/Adam/gradients/loss/dense_4_loss/Mean_grad/Cast*
T0*)
_class
loc:@loss/dense_4_loss/Mean*'
_output_shapes
:���������+
�
;training/Adam/gradients/loss/dense_4_loss/Square_grad/ConstConst<^training/Adam/gradients/loss/dense_4_loss/Mean_grad/truediv*+
_class!
loc:@loss/dense_4_loss/Square*
valueB
 *   @*
dtype0*
_output_shapes
: 
�
9training/Adam/gradients/loss/dense_4_loss/Square_grad/MulMulloss/dense_4_loss/sub;training/Adam/gradients/loss/dense_4_loss/Square_grad/Const*
T0*+
_class!
loc:@loss/dense_4_loss/Square*'
_output_shapes
:���������+
�
;training/Adam/gradients/loss/dense_4_loss/Square_grad/Mul_1Mul;training/Adam/gradients/loss/dense_4_loss/Mean_grad/truediv9training/Adam/gradients/loss/dense_4_loss/Square_grad/Mul*
T0*+
_class!
loc:@loss/dense_4_loss/Square*'
_output_shapes
:���������+
�
8training/Adam/gradients/loss/dense_4_loss/sub_grad/ShapeShapedense_4/Relu*
T0*(
_class
loc:@loss/dense_4_loss/sub*
out_type0*
_output_shapes
:
�
:training/Adam/gradients/loss/dense_4_loss/sub_grad/Shape_1Shapedense_4_target*
T0*(
_class
loc:@loss/dense_4_loss/sub*
out_type0*
_output_shapes
:
�
Htraining/Adam/gradients/loss/dense_4_loss/sub_grad/BroadcastGradientArgsBroadcastGradientArgs8training/Adam/gradients/loss/dense_4_loss/sub_grad/Shape:training/Adam/gradients/loss/dense_4_loss/sub_grad/Shape_1*
T0*(
_class
loc:@loss/dense_4_loss/sub*2
_output_shapes 
:���������:���������
�
6training/Adam/gradients/loss/dense_4_loss/sub_grad/SumSum;training/Adam/gradients/loss/dense_4_loss/Square_grad/Mul_1Htraining/Adam/gradients/loss/dense_4_loss/sub_grad/BroadcastGradientArgs*
T0*(
_class
loc:@loss/dense_4_loss/sub*
_output_shapes
:*
	keep_dims( *

Tidx0
�
:training/Adam/gradients/loss/dense_4_loss/sub_grad/ReshapeReshape6training/Adam/gradients/loss/dense_4_loss/sub_grad/Sum8training/Adam/gradients/loss/dense_4_loss/sub_grad/Shape*
T0*(
_class
loc:@loss/dense_4_loss/sub*
Tshape0*'
_output_shapes
:���������+
�
8training/Adam/gradients/loss/dense_4_loss/sub_grad/Sum_1Sum;training/Adam/gradients/loss/dense_4_loss/Square_grad/Mul_1Jtraining/Adam/gradients/loss/dense_4_loss/sub_grad/BroadcastGradientArgs:1*
T0*(
_class
loc:@loss/dense_4_loss/sub*
_output_shapes
:*
	keep_dims( *

Tidx0
�
6training/Adam/gradients/loss/dense_4_loss/sub_grad/NegNeg8training/Adam/gradients/loss/dense_4_loss/sub_grad/Sum_1*
T0*(
_class
loc:@loss/dense_4_loss/sub*
_output_shapes
:
�
<training/Adam/gradients/loss/dense_4_loss/sub_grad/Reshape_1Reshape6training/Adam/gradients/loss/dense_4_loss/sub_grad/Neg:training/Adam/gradients/loss/dense_4_loss/sub_grad/Shape_1*
T0*(
_class
loc:@loss/dense_4_loss/sub*
Tshape0*0
_output_shapes
:������������������
�
2training/Adam/gradients/dense_4/Relu_grad/ReluGradReluGrad:training/Adam/gradients/loss/dense_4_loss/sub_grad/Reshapedense_4/Relu*
T0*
_class
loc:@dense_4/Relu*'
_output_shapes
:���������+
�
8training/Adam/gradients/dense_4/BiasAdd_grad/BiasAddGradBiasAddGrad2training/Adam/gradients/dense_4/Relu_grad/ReluGrad*
data_formatNHWC*
_output_shapes
:+*
T0*"
_class
loc:@dense_4/BiasAdd
�
2training/Adam/gradients/dense_4/MatMul_grad/MatMulMatMul2training/Adam/gradients/dense_4/Relu_grad/ReluGraddense_4/kernel/read*
T0*!
_class
loc:@dense_4/MatMul*
transpose_a( *'
_output_shapes
:���������*
transpose_b(
�
4training/Adam/gradients/dense_4/MatMul_grad/MatMul_1MatMuldense_3/Tanh2training/Adam/gradients/dense_4/Relu_grad/ReluGrad*
T0*!
_class
loc:@dense_4/MatMul*
transpose_a(*
_output_shapes

:+*
transpose_b( 
�
2training/Adam/gradients/dense_3/Tanh_grad/TanhGradTanhGraddense_3/Tanh2training/Adam/gradients/dense_4/MatMul_grad/MatMul*'
_output_shapes
:���������*
T0*
_class
loc:@dense_3/Tanh
�
8training/Adam/gradients/dense_3/BiasAdd_grad/BiasAddGradBiasAddGrad2training/Adam/gradients/dense_3/Tanh_grad/TanhGrad*
T0*"
_class
loc:@dense_3/BiasAdd*
data_formatNHWC*
_output_shapes
:
�
2training/Adam/gradients/dense_3/MatMul_grad/MatMulMatMul2training/Adam/gradients/dense_3/Tanh_grad/TanhGraddense_3/kernel/read*
transpose_b(*
T0*!
_class
loc:@dense_3/MatMul*
transpose_a( *'
_output_shapes
:���������
�
4training/Adam/gradients/dense_3/MatMul_grad/MatMul_1MatMuldense_2/Relu2training/Adam/gradients/dense_3/Tanh_grad/TanhGrad*
T0*!
_class
loc:@dense_3/MatMul*
transpose_a(*
_output_shapes

:*
transpose_b( 
�
2training/Adam/gradients/dense_2/Relu_grad/ReluGradReluGrad2training/Adam/gradients/dense_3/MatMul_grad/MatMuldense_2/Relu*
T0*
_class
loc:@dense_2/Relu*'
_output_shapes
:���������
�
8training/Adam/gradients/dense_2/BiasAdd_grad/BiasAddGradBiasAddGrad2training/Adam/gradients/dense_2/Relu_grad/ReluGrad*
_output_shapes
:*
T0*"
_class
loc:@dense_2/BiasAdd*
data_formatNHWC
�
2training/Adam/gradients/dense_2/MatMul_grad/MatMulMatMul2training/Adam/gradients/dense_2/Relu_grad/ReluGraddense_2/kernel/read*
transpose_a( *'
_output_shapes
:���������
*
transpose_b(*
T0*!
_class
loc:@dense_2/MatMul
�
4training/Adam/gradients/dense_2/MatMul_grad/MatMul_1MatMuldense_1/Tanh2training/Adam/gradients/dense_2/Relu_grad/ReluGrad*
transpose_b( *
T0*!
_class
loc:@dense_2/MatMul*
transpose_a(*
_output_shapes

:

�
training/Adam/gradients/AddNAddNAtraining/Adam/gradients/dense_1/activity_regularizer/Abs_grad/mul2training/Adam/gradients/dense_2/MatMul_grad/MatMul*
T0*3
_class)
'%loc:@dense_1/activity_regularizer/Abs*
N*'
_output_shapes
:���������

�
2training/Adam/gradients/dense_1/Tanh_grad/TanhGradTanhGraddense_1/Tanhtraining/Adam/gradients/AddN*
T0*
_class
loc:@dense_1/Tanh*'
_output_shapes
:���������

�
8training/Adam/gradients/dense_1/BiasAdd_grad/BiasAddGradBiasAddGrad2training/Adam/gradients/dense_1/Tanh_grad/TanhGrad*
T0*"
_class
loc:@dense_1/BiasAdd*
data_formatNHWC*
_output_shapes
:

�
2training/Adam/gradients/dense_1/MatMul_grad/MatMulMatMul2training/Adam/gradients/dense_1/Tanh_grad/TanhGraddense_1/kernel/read*
transpose_a( *'
_output_shapes
:���������+*
transpose_b(*
T0*!
_class
loc:@dense_1/MatMul
�
4training/Adam/gradients/dense_1/MatMul_grad/MatMul_1MatMulinput_12training/Adam/gradients/dense_1/Tanh_grad/TanhGrad*
transpose_a(*
_output_shapes

:+
*
transpose_b( *
T0*!
_class
loc:@dense_1/MatMul
_
training/Adam/AssignAdd/valueConst*
_output_shapes
: *
value	B	 R*
dtype0	
�
training/Adam/AssignAdd	AssignAddAdam/iterationstraining/Adam/AssignAdd/value*
use_locking( *
T0	*"
_class
loc:@Adam/iterations*
_output_shapes
: 
p
training/Adam/CastCastAdam/iterations/read*
Truncate( *

DstT0*
_output_shapes
: *

SrcT0	
X
training/Adam/add/yConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
b
training/Adam/addAddtraining/Adam/Casttraining/Adam/add/y*
_output_shapes
: *
T0
^
training/Adam/PowPowAdam/beta_2/readtraining/Adam/add*
T0*
_output_shapes
: 
X
training/Adam/sub/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
a
training/Adam/subSubtraining/Adam/sub/xtraining/Adam/Pow*
T0*
_output_shapes
: 
X
training/Adam/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Z
training/Adam/Const_1Const*
valueB
 *  �*
dtype0*
_output_shapes
: 
y
#training/Adam/clip_by_value/MinimumMinimumtraining/Adam/subtraining/Adam/Const_1*
_output_shapes
: *
T0
�
training/Adam/clip_by_valueMaximum#training/Adam/clip_by_value/Minimumtraining/Adam/Const*
T0*
_output_shapes
: 
X
training/Adam/SqrtSqrttraining/Adam/clip_by_value*
_output_shapes
: *
T0
`
training/Adam/Pow_1PowAdam/beta_1/readtraining/Adam/add*
T0*
_output_shapes
: 
Z
training/Adam/sub_1/xConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
g
training/Adam/sub_1Subtraining/Adam/sub_1/xtraining/Adam/Pow_1*
_output_shapes
: *
T0
j
training/Adam/truedivRealDivtraining/Adam/Sqrttraining/Adam/sub_1*
_output_shapes
: *
T0
^
training/Adam/mulMulAdam/lr/readtraining/Adam/truediv*
T0*
_output_shapes
: 
h
training/Adam/zerosConst*
valueB+
*    *
dtype0*
_output_shapes

:+

�
training/Adam/Variable
VariableV2*
	container *
_output_shapes

:+
*
shape
:+
*
shared_name *
dtype0
�
training/Adam/Variable/AssignAssigntraining/Adam/Variabletraining/Adam/zeros*
use_locking(*
T0*)
_class
loc:@training/Adam/Variable*
validate_shape(*
_output_shapes

:+

�
training/Adam/Variable/readIdentitytraining/Adam/Variable*)
_class
loc:@training/Adam/Variable*
_output_shapes

:+
*
T0
b
training/Adam/zeros_1Const*
dtype0*
_output_shapes
:
*
valueB
*    
�
training/Adam/Variable_1
VariableV2*
	container *
_output_shapes
:
*
shape:
*
shared_name *
dtype0
�
training/Adam/Variable_1/AssignAssigntraining/Adam/Variable_1training/Adam/zeros_1*
validate_shape(*
_output_shapes
:
*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_1
�
training/Adam/Variable_1/readIdentitytraining/Adam/Variable_1*
T0*+
_class!
loc:@training/Adam/Variable_1*
_output_shapes
:

j
training/Adam/zeros_2Const*
valueB
*    *
dtype0*
_output_shapes

:

�
training/Adam/Variable_2
VariableV2*
shape
:
*
shared_name *
dtype0*
	container *
_output_shapes

:

�
training/Adam/Variable_2/AssignAssigntraining/Adam/Variable_2training/Adam/zeros_2*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_2*
validate_shape(*
_output_shapes

:

�
training/Adam/Variable_2/readIdentitytraining/Adam/Variable_2*
T0*+
_class!
loc:@training/Adam/Variable_2*
_output_shapes

:

b
training/Adam/zeros_3Const*
valueB*    *
dtype0*
_output_shapes
:
�
training/Adam/Variable_3
VariableV2*
dtype0*
	container *
_output_shapes
:*
shape:*
shared_name 
�
training/Adam/Variable_3/AssignAssigntraining/Adam/Variable_3training/Adam/zeros_3*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_3*
validate_shape(*
_output_shapes
:
�
training/Adam/Variable_3/readIdentitytraining/Adam/Variable_3*
T0*+
_class!
loc:@training/Adam/Variable_3*
_output_shapes
:
j
training/Adam/zeros_4Const*
valueB*    *
dtype0*
_output_shapes

:
�
training/Adam/Variable_4
VariableV2*
shape
:*
shared_name *
dtype0*
	container *
_output_shapes

:
�
training/Adam/Variable_4/AssignAssigntraining/Adam/Variable_4training/Adam/zeros_4*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_4*
validate_shape(*
_output_shapes

:
�
training/Adam/Variable_4/readIdentitytraining/Adam/Variable_4*
_output_shapes

:*
T0*+
_class!
loc:@training/Adam/Variable_4
b
training/Adam/zeros_5Const*
valueB*    *
dtype0*
_output_shapes
:
�
training/Adam/Variable_5
VariableV2*
shared_name *
dtype0*
	container *
_output_shapes
:*
shape:
�
training/Adam/Variable_5/AssignAssigntraining/Adam/Variable_5training/Adam/zeros_5*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_5*
validate_shape(*
_output_shapes
:
�
training/Adam/Variable_5/readIdentitytraining/Adam/Variable_5*
T0*+
_class!
loc:@training/Adam/Variable_5*
_output_shapes
:
j
training/Adam/zeros_6Const*
valueB+*    *
dtype0*
_output_shapes

:+
�
training/Adam/Variable_6
VariableV2*
	container *
_output_shapes

:+*
shape
:+*
shared_name *
dtype0
�
training/Adam/Variable_6/AssignAssigntraining/Adam/Variable_6training/Adam/zeros_6*+
_class!
loc:@training/Adam/Variable_6*
validate_shape(*
_output_shapes

:+*
use_locking(*
T0
�
training/Adam/Variable_6/readIdentitytraining/Adam/Variable_6*
_output_shapes

:+*
T0*+
_class!
loc:@training/Adam/Variable_6
b
training/Adam/zeros_7Const*
valueB+*    *
dtype0*
_output_shapes
:+
�
training/Adam/Variable_7
VariableV2*
shared_name *
dtype0*
	container *
_output_shapes
:+*
shape:+
�
training/Adam/Variable_7/AssignAssigntraining/Adam/Variable_7training/Adam/zeros_7*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_7*
validate_shape(*
_output_shapes
:+
�
training/Adam/Variable_7/readIdentitytraining/Adam/Variable_7*
T0*+
_class!
loc:@training/Adam/Variable_7*
_output_shapes
:+
j
training/Adam/zeros_8Const*
valueB+
*    *
dtype0*
_output_shapes

:+

�
training/Adam/Variable_8
VariableV2*
shape
:+
*
shared_name *
dtype0*
	container *
_output_shapes

:+

�
training/Adam/Variable_8/AssignAssigntraining/Adam/Variable_8training/Adam/zeros_8*
T0*+
_class!
loc:@training/Adam/Variable_8*
validate_shape(*
_output_shapes

:+
*
use_locking(
�
training/Adam/Variable_8/readIdentitytraining/Adam/Variable_8*
_output_shapes

:+
*
T0*+
_class!
loc:@training/Adam/Variable_8
b
training/Adam/zeros_9Const*
valueB
*    *
dtype0*
_output_shapes
:

�
training/Adam/Variable_9
VariableV2*
shape:
*
shared_name *
dtype0*
	container *
_output_shapes
:

�
training/Adam/Variable_9/AssignAssigntraining/Adam/Variable_9training/Adam/zeros_9*+
_class!
loc:@training/Adam/Variable_9*
validate_shape(*
_output_shapes
:
*
use_locking(*
T0
�
training/Adam/Variable_9/readIdentitytraining/Adam/Variable_9*
_output_shapes
:
*
T0*+
_class!
loc:@training/Adam/Variable_9
k
training/Adam/zeros_10Const*
valueB
*    *
dtype0*
_output_shapes

:

�
training/Adam/Variable_10
VariableV2*
shape
:
*
shared_name *
dtype0*
	container *
_output_shapes

:

�
 training/Adam/Variable_10/AssignAssigntraining/Adam/Variable_10training/Adam/zeros_10*
T0*,
_class"
 loc:@training/Adam/Variable_10*
validate_shape(*
_output_shapes

:
*
use_locking(
�
training/Adam/Variable_10/readIdentitytraining/Adam/Variable_10*
T0*,
_class"
 loc:@training/Adam/Variable_10*
_output_shapes

:

c
training/Adam/zeros_11Const*
dtype0*
_output_shapes
:*
valueB*    
�
training/Adam/Variable_11
VariableV2*
dtype0*
	container *
_output_shapes
:*
shape:*
shared_name 
�
 training/Adam/Variable_11/AssignAssigntraining/Adam/Variable_11training/Adam/zeros_11*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_11*
validate_shape(*
_output_shapes
:
�
training/Adam/Variable_11/readIdentitytraining/Adam/Variable_11*
T0*,
_class"
 loc:@training/Adam/Variable_11*
_output_shapes
:
k
training/Adam/zeros_12Const*
dtype0*
_output_shapes

:*
valueB*    
�
training/Adam/Variable_12
VariableV2*
shape
:*
shared_name *
dtype0*
	container *
_output_shapes

:
�
 training/Adam/Variable_12/AssignAssigntraining/Adam/Variable_12training/Adam/zeros_12*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_12*
validate_shape(*
_output_shapes

:
�
training/Adam/Variable_12/readIdentitytraining/Adam/Variable_12*
T0*,
_class"
 loc:@training/Adam/Variable_12*
_output_shapes

:
c
training/Adam/zeros_13Const*
valueB*    *
dtype0*
_output_shapes
:
�
training/Adam/Variable_13
VariableV2*
shape:*
shared_name *
dtype0*
	container *
_output_shapes
:
�
 training/Adam/Variable_13/AssignAssigntraining/Adam/Variable_13training/Adam/zeros_13*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_13*
validate_shape(*
_output_shapes
:
�
training/Adam/Variable_13/readIdentitytraining/Adam/Variable_13*
T0*,
_class"
 loc:@training/Adam/Variable_13*
_output_shapes
:
k
training/Adam/zeros_14Const*
valueB+*    *
dtype0*
_output_shapes

:+
�
training/Adam/Variable_14
VariableV2*
shared_name *
dtype0*
	container *
_output_shapes

:+*
shape
:+
�
 training/Adam/Variable_14/AssignAssigntraining/Adam/Variable_14training/Adam/zeros_14*
T0*,
_class"
 loc:@training/Adam/Variable_14*
validate_shape(*
_output_shapes

:+*
use_locking(
�
training/Adam/Variable_14/readIdentitytraining/Adam/Variable_14*
T0*,
_class"
 loc:@training/Adam/Variable_14*
_output_shapes

:+
c
training/Adam/zeros_15Const*
valueB+*    *
dtype0*
_output_shapes
:+
�
training/Adam/Variable_15
VariableV2*
dtype0*
	container *
_output_shapes
:+*
shape:+*
shared_name 
�
 training/Adam/Variable_15/AssignAssigntraining/Adam/Variable_15training/Adam/zeros_15*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_15*
validate_shape(*
_output_shapes
:+
�
training/Adam/Variable_15/readIdentitytraining/Adam/Variable_15*
_output_shapes
:+*
T0*,
_class"
 loc:@training/Adam/Variable_15
p
&training/Adam/zeros_16/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
a
training/Adam/zeros_16/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
training/Adam/zeros_16Fill&training/Adam/zeros_16/shape_as_tensortraining/Adam/zeros_16/Const*
_output_shapes
:*
T0*

index_type0
�
training/Adam/Variable_16
VariableV2*
shape:*
shared_name *
dtype0*
	container *
_output_shapes
:
�
 training/Adam/Variable_16/AssignAssigntraining/Adam/Variable_16training/Adam/zeros_16*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_16*
validate_shape(*
_output_shapes
:
�
training/Adam/Variable_16/readIdentitytraining/Adam/Variable_16*,
_class"
 loc:@training/Adam/Variable_16*
_output_shapes
:*
T0
p
&training/Adam/zeros_17/shape_as_tensorConst*
_output_shapes
:*
valueB:*
dtype0
a
training/Adam/zeros_17/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
training/Adam/zeros_17Fill&training/Adam/zeros_17/shape_as_tensortraining/Adam/zeros_17/Const*

index_type0*
_output_shapes
:*
T0
�
training/Adam/Variable_17
VariableV2*
dtype0*
	container *
_output_shapes
:*
shape:*
shared_name 
�
 training/Adam/Variable_17/AssignAssigntraining/Adam/Variable_17training/Adam/zeros_17*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_17*
validate_shape(*
_output_shapes
:
�
training/Adam/Variable_17/readIdentitytraining/Adam/Variable_17*
T0*,
_class"
 loc:@training/Adam/Variable_17*
_output_shapes
:
p
&training/Adam/zeros_18/shape_as_tensorConst*
_output_shapes
:*
valueB:*
dtype0
a
training/Adam/zeros_18/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
training/Adam/zeros_18Fill&training/Adam/zeros_18/shape_as_tensortraining/Adam/zeros_18/Const*
T0*

index_type0*
_output_shapes
:
�
training/Adam/Variable_18
VariableV2*
shared_name *
dtype0*
	container *
_output_shapes
:*
shape:
�
 training/Adam/Variable_18/AssignAssigntraining/Adam/Variable_18training/Adam/zeros_18*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_18*
validate_shape(*
_output_shapes
:
�
training/Adam/Variable_18/readIdentitytraining/Adam/Variable_18*
T0*,
_class"
 loc:@training/Adam/Variable_18*
_output_shapes
:
p
&training/Adam/zeros_19/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
a
training/Adam/zeros_19/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
training/Adam/zeros_19Fill&training/Adam/zeros_19/shape_as_tensortraining/Adam/zeros_19/Const*
_output_shapes
:*
T0*

index_type0
�
training/Adam/Variable_19
VariableV2*
dtype0*
	container *
_output_shapes
:*
shape:*
shared_name 
�
 training/Adam/Variable_19/AssignAssigntraining/Adam/Variable_19training/Adam/zeros_19*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_19*
validate_shape(*
_output_shapes
:
�
training/Adam/Variable_19/readIdentitytraining/Adam/Variable_19*
T0*,
_class"
 loc:@training/Adam/Variable_19*
_output_shapes
:
p
&training/Adam/zeros_20/shape_as_tensorConst*
_output_shapes
:*
valueB:*
dtype0
a
training/Adam/zeros_20/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    
�
training/Adam/zeros_20Fill&training/Adam/zeros_20/shape_as_tensortraining/Adam/zeros_20/Const*
_output_shapes
:*
T0*

index_type0
�
training/Adam/Variable_20
VariableV2*
shared_name *
dtype0*
	container *
_output_shapes
:*
shape:
�
 training/Adam/Variable_20/AssignAssigntraining/Adam/Variable_20training/Adam/zeros_20*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_20*
validate_shape(*
_output_shapes
:
�
training/Adam/Variable_20/readIdentitytraining/Adam/Variable_20*
T0*,
_class"
 loc:@training/Adam/Variable_20*
_output_shapes
:
p
&training/Adam/zeros_21/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
a
training/Adam/zeros_21/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
training/Adam/zeros_21Fill&training/Adam/zeros_21/shape_as_tensortraining/Adam/zeros_21/Const*

index_type0*
_output_shapes
:*
T0
�
training/Adam/Variable_21
VariableV2*
shape:*
shared_name *
dtype0*
	container *
_output_shapes
:
�
 training/Adam/Variable_21/AssignAssigntraining/Adam/Variable_21training/Adam/zeros_21*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_21
�
training/Adam/Variable_21/readIdentitytraining/Adam/Variable_21*
T0*,
_class"
 loc:@training/Adam/Variable_21*
_output_shapes
:
p
&training/Adam/zeros_22/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
a
training/Adam/zeros_22/ConstConst*
_output_shapes
: *
valueB
 *    *
dtype0
�
training/Adam/zeros_22Fill&training/Adam/zeros_22/shape_as_tensortraining/Adam/zeros_22/Const*
T0*

index_type0*
_output_shapes
:
�
training/Adam/Variable_22
VariableV2*
dtype0*
	container *
_output_shapes
:*
shape:*
shared_name 
�
 training/Adam/Variable_22/AssignAssigntraining/Adam/Variable_22training/Adam/zeros_22*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_22*
validate_shape(*
_output_shapes
:
�
training/Adam/Variable_22/readIdentitytraining/Adam/Variable_22*
_output_shapes
:*
T0*,
_class"
 loc:@training/Adam/Variable_22
p
&training/Adam/zeros_23/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
a
training/Adam/zeros_23/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
training/Adam/zeros_23Fill&training/Adam/zeros_23/shape_as_tensortraining/Adam/zeros_23/Const*
_output_shapes
:*
T0*

index_type0
�
training/Adam/Variable_23
VariableV2*
dtype0*
	container *
_output_shapes
:*
shape:*
shared_name 
�
 training/Adam/Variable_23/AssignAssigntraining/Adam/Variable_23training/Adam/zeros_23*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_23*
validate_shape(*
_output_shapes
:
�
training/Adam/Variable_23/readIdentitytraining/Adam/Variable_23*
T0*,
_class"
 loc:@training/Adam/Variable_23*
_output_shapes
:
r
training/Adam/mul_1MulAdam/beta_1/readtraining/Adam/Variable/read*
T0*
_output_shapes

:+

Z
training/Adam/sub_2/xConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
d
training/Adam/sub_2Subtraining/Adam/sub_2/xAdam/beta_1/read*
T0*
_output_shapes
: 
�
training/Adam/mul_2Multraining/Adam/sub_24training/Adam/gradients/dense_1/MatMul_grad/MatMul_1*
T0*
_output_shapes

:+

m
training/Adam/add_1Addtraining/Adam/mul_1training/Adam/mul_2*
_output_shapes

:+
*
T0
t
training/Adam/mul_3MulAdam/beta_2/readtraining/Adam/Variable_8/read*
T0*
_output_shapes

:+

Z
training/Adam/sub_3/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
d
training/Adam/sub_3Subtraining/Adam/sub_3/xAdam/beta_2/read*
T0*
_output_shapes
: 
}
training/Adam/SquareSquare4training/Adam/gradients/dense_1/MatMul_grad/MatMul_1*
T0*
_output_shapes

:+

n
training/Adam/mul_4Multraining/Adam/sub_3training/Adam/Square*
_output_shapes

:+
*
T0
m
training/Adam/add_2Addtraining/Adam/mul_3training/Adam/mul_4*
_output_shapes

:+
*
T0
k
training/Adam/mul_5Multraining/Adam/multraining/Adam/add_1*
T0*
_output_shapes

:+

Z
training/Adam/Const_2Const*
valueB
 *    *
dtype0*
_output_shapes
: 
Z
training/Adam/Const_3Const*
valueB
 *  �*
dtype0*
_output_shapes
: 
�
%training/Adam/clip_by_value_1/MinimumMinimumtraining/Adam/add_2training/Adam/Const_3*
T0*
_output_shapes

:+

�
training/Adam/clip_by_value_1Maximum%training/Adam/clip_by_value_1/Minimumtraining/Adam/Const_2*
T0*
_output_shapes

:+

d
training/Adam/Sqrt_1Sqrttraining/Adam/clip_by_value_1*
T0*
_output_shapes

:+

Z
training/Adam/add_3/yConst*
valueB
 *���3*
dtype0*
_output_shapes
: 
p
training/Adam/add_3Addtraining/Adam/Sqrt_1training/Adam/add_3/y*
T0*
_output_shapes

:+

u
training/Adam/truediv_1RealDivtraining/Adam/mul_5training/Adam/add_3*
_output_shapes

:+
*
T0
q
training/Adam/sub_4Subdense_1/kernel/readtraining/Adam/truediv_1*
T0*
_output_shapes

:+

�
training/Adam/AssignAssigntraining/Adam/Variabletraining/Adam/add_1*
use_locking(*
T0*)
_class
loc:@training/Adam/Variable*
validate_shape(*
_output_shapes

:+

�
training/Adam/Assign_1Assigntraining/Adam/Variable_8training/Adam/add_2*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_8*
validate_shape(*
_output_shapes

:+

�
training/Adam/Assign_2Assigndense_1/kerneltraining/Adam/sub_4*
T0*!
_class
loc:@dense_1/kernel*
validate_shape(*
_output_shapes

:+
*
use_locking(
p
training/Adam/mul_6MulAdam/beta_1/readtraining/Adam/Variable_1/read*
T0*
_output_shapes
:

Z
training/Adam/sub_5/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
d
training/Adam/sub_5Subtraining/Adam/sub_5/xAdam/beta_1/read*
T0*
_output_shapes
: 
�
training/Adam/mul_7Multraining/Adam/sub_58training/Adam/gradients/dense_1/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:

i
training/Adam/add_4Addtraining/Adam/mul_6training/Adam/mul_7*
_output_shapes
:
*
T0
p
training/Adam/mul_8MulAdam/beta_2/readtraining/Adam/Variable_9/read*
T0*
_output_shapes
:

Z
training/Adam/sub_6/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
d
training/Adam/sub_6Subtraining/Adam/sub_6/xAdam/beta_2/read*
_output_shapes
: *
T0

training/Adam/Square_1Square8training/Adam/gradients/dense_1/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:

l
training/Adam/mul_9Multraining/Adam/sub_6training/Adam/Square_1*
_output_shapes
:
*
T0
i
training/Adam/add_5Addtraining/Adam/mul_8training/Adam/mul_9*
_output_shapes
:
*
T0
h
training/Adam/mul_10Multraining/Adam/multraining/Adam/add_4*
T0*
_output_shapes
:

Z
training/Adam/Const_4Const*
valueB
 *    *
dtype0*
_output_shapes
: 
Z
training/Adam/Const_5Const*
valueB
 *  �*
dtype0*
_output_shapes
: 
�
%training/Adam/clip_by_value_2/MinimumMinimumtraining/Adam/add_5training/Adam/Const_5*
T0*
_output_shapes
:

�
training/Adam/clip_by_value_2Maximum%training/Adam/clip_by_value_2/Minimumtraining/Adam/Const_4*
_output_shapes
:
*
T0
`
training/Adam/Sqrt_2Sqrttraining/Adam/clip_by_value_2*
T0*
_output_shapes
:

Z
training/Adam/add_6/yConst*
valueB
 *���3*
dtype0*
_output_shapes
: 
l
training/Adam/add_6Addtraining/Adam/Sqrt_2training/Adam/add_6/y*
_output_shapes
:
*
T0
r
training/Adam/truediv_2RealDivtraining/Adam/mul_10training/Adam/add_6*
T0*
_output_shapes
:

k
training/Adam/sub_7Subdense_1/bias/readtraining/Adam/truediv_2*
T0*
_output_shapes
:

�
training/Adam/Assign_3Assigntraining/Adam/Variable_1training/Adam/add_4*+
_class!
loc:@training/Adam/Variable_1*
validate_shape(*
_output_shapes
:
*
use_locking(*
T0
�
training/Adam/Assign_4Assigntraining/Adam/Variable_9training/Adam/add_5*
validate_shape(*
_output_shapes
:
*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_9
�
training/Adam/Assign_5Assigndense_1/biastraining/Adam/sub_7*
use_locking(*
T0*
_class
loc:@dense_1/bias*
validate_shape(*
_output_shapes
:

u
training/Adam/mul_11MulAdam/beta_1/readtraining/Adam/Variable_2/read*
T0*
_output_shapes

:

Z
training/Adam/sub_8/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
d
training/Adam/sub_8Subtraining/Adam/sub_8/xAdam/beta_1/read*
T0*
_output_shapes
: 
�
training/Adam/mul_12Multraining/Adam/sub_84training/Adam/gradients/dense_2/MatMul_grad/MatMul_1*
T0*
_output_shapes

:

o
training/Adam/add_7Addtraining/Adam/mul_11training/Adam/mul_12*
_output_shapes

:
*
T0
v
training/Adam/mul_13MulAdam/beta_2/readtraining/Adam/Variable_10/read*
T0*
_output_shapes

:

Z
training/Adam/sub_9/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
d
training/Adam/sub_9Subtraining/Adam/sub_9/xAdam/beta_2/read*
T0*
_output_shapes
: 

training/Adam/Square_2Square4training/Adam/gradients/dense_2/MatMul_grad/MatMul_1*
T0*
_output_shapes

:

q
training/Adam/mul_14Multraining/Adam/sub_9training/Adam/Square_2*
T0*
_output_shapes

:

o
training/Adam/add_8Addtraining/Adam/mul_13training/Adam/mul_14*
T0*
_output_shapes

:

l
training/Adam/mul_15Multraining/Adam/multraining/Adam/add_7*
_output_shapes

:
*
T0
Z
training/Adam/Const_6Const*
valueB
 *    *
dtype0*
_output_shapes
: 
Z
training/Adam/Const_7Const*
valueB
 *  �*
dtype0*
_output_shapes
: 
�
%training/Adam/clip_by_value_3/MinimumMinimumtraining/Adam/add_8training/Adam/Const_7*
T0*
_output_shapes

:

�
training/Adam/clip_by_value_3Maximum%training/Adam/clip_by_value_3/Minimumtraining/Adam/Const_6*
T0*
_output_shapes

:

d
training/Adam/Sqrt_3Sqrttraining/Adam/clip_by_value_3*
T0*
_output_shapes

:

Z
training/Adam/add_9/yConst*
valueB
 *���3*
dtype0*
_output_shapes
: 
p
training/Adam/add_9Addtraining/Adam/Sqrt_3training/Adam/add_9/y*
T0*
_output_shapes

:

v
training/Adam/truediv_3RealDivtraining/Adam/mul_15training/Adam/add_9*
T0*
_output_shapes

:

r
training/Adam/sub_10Subdense_2/kernel/readtraining/Adam/truediv_3*
T0*
_output_shapes

:

�
training/Adam/Assign_6Assigntraining/Adam/Variable_2training/Adam/add_7*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_2*
validate_shape(*
_output_shapes

:

�
training/Adam/Assign_7Assigntraining/Adam/Variable_10training/Adam/add_8*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_10*
validate_shape(*
_output_shapes

:

�
training/Adam/Assign_8Assigndense_2/kerneltraining/Adam/sub_10*
validate_shape(*
_output_shapes

:
*
use_locking(*
T0*!
_class
loc:@dense_2/kernel
q
training/Adam/mul_16MulAdam/beta_1/readtraining/Adam/Variable_3/read*
T0*
_output_shapes
:
[
training/Adam/sub_11/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_11Subtraining/Adam/sub_11/xAdam/beta_1/read*
T0*
_output_shapes
: 
�
training/Adam/mul_17Multraining/Adam/sub_118training/Adam/gradients/dense_2/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:
l
training/Adam/add_10Addtraining/Adam/mul_16training/Adam/mul_17*
T0*
_output_shapes
:
r
training/Adam/mul_18MulAdam/beta_2/readtraining/Adam/Variable_11/read*
T0*
_output_shapes
:
[
training/Adam/sub_12/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_12Subtraining/Adam/sub_12/xAdam/beta_2/read*
T0*
_output_shapes
: 

training/Adam/Square_3Square8training/Adam/gradients/dense_2/BiasAdd_grad/BiasAddGrad*
_output_shapes
:*
T0
n
training/Adam/mul_19Multraining/Adam/sub_12training/Adam/Square_3*
T0*
_output_shapes
:
l
training/Adam/add_11Addtraining/Adam/mul_18training/Adam/mul_19*
T0*
_output_shapes
:
i
training/Adam/mul_20Multraining/Adam/multraining/Adam/add_10*
T0*
_output_shapes
:
Z
training/Adam/Const_8Const*
valueB
 *    *
dtype0*
_output_shapes
: 
Z
training/Adam/Const_9Const*
valueB
 *  �*
dtype0*
_output_shapes
: 
�
%training/Adam/clip_by_value_4/MinimumMinimumtraining/Adam/add_11training/Adam/Const_9*
T0*
_output_shapes
:
�
training/Adam/clip_by_value_4Maximum%training/Adam/clip_by_value_4/Minimumtraining/Adam/Const_8*
T0*
_output_shapes
:
`
training/Adam/Sqrt_4Sqrttraining/Adam/clip_by_value_4*
_output_shapes
:*
T0
[
training/Adam/add_12/yConst*
valueB
 *���3*
dtype0*
_output_shapes
: 
n
training/Adam/add_12Addtraining/Adam/Sqrt_4training/Adam/add_12/y*
T0*
_output_shapes
:
s
training/Adam/truediv_4RealDivtraining/Adam/mul_20training/Adam/add_12*
T0*
_output_shapes
:
l
training/Adam/sub_13Subdense_2/bias/readtraining/Adam/truediv_4*
T0*
_output_shapes
:
�
training/Adam/Assign_9Assigntraining/Adam/Variable_3training/Adam/add_10*
T0*+
_class!
loc:@training/Adam/Variable_3*
validate_shape(*
_output_shapes
:*
use_locking(
�
training/Adam/Assign_10Assigntraining/Adam/Variable_11training/Adam/add_11*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_11*
validate_shape(*
_output_shapes
:
�
training/Adam/Assign_11Assigndense_2/biastraining/Adam/sub_13*
use_locking(*
T0*
_class
loc:@dense_2/bias*
validate_shape(*
_output_shapes
:
u
training/Adam/mul_21MulAdam/beta_1/readtraining/Adam/Variable_4/read*
T0*
_output_shapes

:
[
training/Adam/sub_14/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_14Subtraining/Adam/sub_14/xAdam/beta_1/read*
T0*
_output_shapes
: 
�
training/Adam/mul_22Multraining/Adam/sub_144training/Adam/gradients/dense_3/MatMul_grad/MatMul_1*
_output_shapes

:*
T0
p
training/Adam/add_13Addtraining/Adam/mul_21training/Adam/mul_22*
T0*
_output_shapes

:
v
training/Adam/mul_23MulAdam/beta_2/readtraining/Adam/Variable_12/read*
_output_shapes

:*
T0
[
training/Adam/sub_15/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_15Subtraining/Adam/sub_15/xAdam/beta_2/read*
T0*
_output_shapes
: 

training/Adam/Square_4Square4training/Adam/gradients/dense_3/MatMul_grad/MatMul_1*
T0*
_output_shapes

:
r
training/Adam/mul_24Multraining/Adam/sub_15training/Adam/Square_4*
_output_shapes

:*
T0
p
training/Adam/add_14Addtraining/Adam/mul_23training/Adam/mul_24*
T0*
_output_shapes

:
m
training/Adam/mul_25Multraining/Adam/multraining/Adam/add_13*
T0*
_output_shapes

:
[
training/Adam/Const_10Const*
valueB
 *    *
dtype0*
_output_shapes
: 
[
training/Adam/Const_11Const*
dtype0*
_output_shapes
: *
valueB
 *  �
�
%training/Adam/clip_by_value_5/MinimumMinimumtraining/Adam/add_14training/Adam/Const_11*
T0*
_output_shapes

:
�
training/Adam/clip_by_value_5Maximum%training/Adam/clip_by_value_5/Minimumtraining/Adam/Const_10*
T0*
_output_shapes

:
d
training/Adam/Sqrt_5Sqrttraining/Adam/clip_by_value_5*
T0*
_output_shapes

:
[
training/Adam/add_15/yConst*
_output_shapes
: *
valueB
 *���3*
dtype0
r
training/Adam/add_15Addtraining/Adam/Sqrt_5training/Adam/add_15/y*
_output_shapes

:*
T0
w
training/Adam/truediv_5RealDivtraining/Adam/mul_25training/Adam/add_15*
T0*
_output_shapes

:
r
training/Adam/sub_16Subdense_3/kernel/readtraining/Adam/truediv_5*
T0*
_output_shapes

:
�
training/Adam/Assign_12Assigntraining/Adam/Variable_4training/Adam/add_13*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_4*
validate_shape(*
_output_shapes

:
�
training/Adam/Assign_13Assigntraining/Adam/Variable_12training/Adam/add_14*
_output_shapes

:*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_12*
validate_shape(
�
training/Adam/Assign_14Assigndense_3/kerneltraining/Adam/sub_16*
use_locking(*
T0*!
_class
loc:@dense_3/kernel*
validate_shape(*
_output_shapes

:
q
training/Adam/mul_26MulAdam/beta_1/readtraining/Adam/Variable_5/read*
T0*
_output_shapes
:
[
training/Adam/sub_17/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_17Subtraining/Adam/sub_17/xAdam/beta_1/read*
T0*
_output_shapes
: 
�
training/Adam/mul_27Multraining/Adam/sub_178training/Adam/gradients/dense_3/BiasAdd_grad/BiasAddGrad*
_output_shapes
:*
T0
l
training/Adam/add_16Addtraining/Adam/mul_26training/Adam/mul_27*
T0*
_output_shapes
:
r
training/Adam/mul_28MulAdam/beta_2/readtraining/Adam/Variable_13/read*
T0*
_output_shapes
:
[
training/Adam/sub_18/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_18Subtraining/Adam/sub_18/xAdam/beta_2/read*
T0*
_output_shapes
: 

training/Adam/Square_5Square8training/Adam/gradients/dense_3/BiasAdd_grad/BiasAddGrad*
_output_shapes
:*
T0
n
training/Adam/mul_29Multraining/Adam/sub_18training/Adam/Square_5*
T0*
_output_shapes
:
l
training/Adam/add_17Addtraining/Adam/mul_28training/Adam/mul_29*
T0*
_output_shapes
:
i
training/Adam/mul_30Multraining/Adam/multraining/Adam/add_16*
T0*
_output_shapes
:
[
training/Adam/Const_12Const*
valueB
 *    *
dtype0*
_output_shapes
: 
[
training/Adam/Const_13Const*
dtype0*
_output_shapes
: *
valueB
 *  �
�
%training/Adam/clip_by_value_6/MinimumMinimumtraining/Adam/add_17training/Adam/Const_13*
T0*
_output_shapes
:
�
training/Adam/clip_by_value_6Maximum%training/Adam/clip_by_value_6/Minimumtraining/Adam/Const_12*
_output_shapes
:*
T0
`
training/Adam/Sqrt_6Sqrttraining/Adam/clip_by_value_6*
_output_shapes
:*
T0
[
training/Adam/add_18/yConst*
valueB
 *���3*
dtype0*
_output_shapes
: 
n
training/Adam/add_18Addtraining/Adam/Sqrt_6training/Adam/add_18/y*
_output_shapes
:*
T0
s
training/Adam/truediv_6RealDivtraining/Adam/mul_30training/Adam/add_18*
T0*
_output_shapes
:
l
training/Adam/sub_19Subdense_3/bias/readtraining/Adam/truediv_6*
_output_shapes
:*
T0
�
training/Adam/Assign_15Assigntraining/Adam/Variable_5training/Adam/add_16*
_output_shapes
:*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_5*
validate_shape(
�
training/Adam/Assign_16Assigntraining/Adam/Variable_13training/Adam/add_17*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_13
�
training/Adam/Assign_17Assigndense_3/biastraining/Adam/sub_19*
T0*
_class
loc:@dense_3/bias*
validate_shape(*
_output_shapes
:*
use_locking(
u
training/Adam/mul_31MulAdam/beta_1/readtraining/Adam/Variable_6/read*
_output_shapes

:+*
T0
[
training/Adam/sub_20/xConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
f
training/Adam/sub_20Subtraining/Adam/sub_20/xAdam/beta_1/read*
T0*
_output_shapes
: 
�
training/Adam/mul_32Multraining/Adam/sub_204training/Adam/gradients/dense_4/MatMul_grad/MatMul_1*
T0*
_output_shapes

:+
p
training/Adam/add_19Addtraining/Adam/mul_31training/Adam/mul_32*
T0*
_output_shapes

:+
v
training/Adam/mul_33MulAdam/beta_2/readtraining/Adam/Variable_14/read*
T0*
_output_shapes

:+
[
training/Adam/sub_21/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_21Subtraining/Adam/sub_21/xAdam/beta_2/read*
T0*
_output_shapes
: 

training/Adam/Square_6Square4training/Adam/gradients/dense_4/MatMul_grad/MatMul_1*
T0*
_output_shapes

:+
r
training/Adam/mul_34Multraining/Adam/sub_21training/Adam/Square_6*
T0*
_output_shapes

:+
p
training/Adam/add_20Addtraining/Adam/mul_33training/Adam/mul_34*
T0*
_output_shapes

:+
m
training/Adam/mul_35Multraining/Adam/multraining/Adam/add_19*
_output_shapes

:+*
T0
[
training/Adam/Const_14Const*
valueB
 *    *
dtype0*
_output_shapes
: 
[
training/Adam/Const_15Const*
valueB
 *  �*
dtype0*
_output_shapes
: 
�
%training/Adam/clip_by_value_7/MinimumMinimumtraining/Adam/add_20training/Adam/Const_15*
T0*
_output_shapes

:+
�
training/Adam/clip_by_value_7Maximum%training/Adam/clip_by_value_7/Minimumtraining/Adam/Const_14*
_output_shapes

:+*
T0
d
training/Adam/Sqrt_7Sqrttraining/Adam/clip_by_value_7*
_output_shapes

:+*
T0
[
training/Adam/add_21/yConst*
valueB
 *���3*
dtype0*
_output_shapes
: 
r
training/Adam/add_21Addtraining/Adam/Sqrt_7training/Adam/add_21/y*
_output_shapes

:+*
T0
w
training/Adam/truediv_7RealDivtraining/Adam/mul_35training/Adam/add_21*
_output_shapes

:+*
T0
r
training/Adam/sub_22Subdense_4/kernel/readtraining/Adam/truediv_7*
_output_shapes

:+*
T0
�
training/Adam/Assign_18Assigntraining/Adam/Variable_6training/Adam/add_19*+
_class!
loc:@training/Adam/Variable_6*
validate_shape(*
_output_shapes

:+*
use_locking(*
T0
�
training/Adam/Assign_19Assigntraining/Adam/Variable_14training/Adam/add_20*
T0*,
_class"
 loc:@training/Adam/Variable_14*
validate_shape(*
_output_shapes

:+*
use_locking(
�
training/Adam/Assign_20Assigndense_4/kerneltraining/Adam/sub_22*
_output_shapes

:+*
use_locking(*
T0*!
_class
loc:@dense_4/kernel*
validate_shape(
q
training/Adam/mul_36MulAdam/beta_1/readtraining/Adam/Variable_7/read*
_output_shapes
:+*
T0
[
training/Adam/sub_23/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_23Subtraining/Adam/sub_23/xAdam/beta_1/read*
_output_shapes
: *
T0
�
training/Adam/mul_37Multraining/Adam/sub_238training/Adam/gradients/dense_4/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:+
l
training/Adam/add_22Addtraining/Adam/mul_36training/Adam/mul_37*
T0*
_output_shapes
:+
r
training/Adam/mul_38MulAdam/beta_2/readtraining/Adam/Variable_15/read*
T0*
_output_shapes
:+
[
training/Adam/sub_24/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_24Subtraining/Adam/sub_24/xAdam/beta_2/read*
T0*
_output_shapes
: 

training/Adam/Square_7Square8training/Adam/gradients/dense_4/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:+
n
training/Adam/mul_39Multraining/Adam/sub_24training/Adam/Square_7*
T0*
_output_shapes
:+
l
training/Adam/add_23Addtraining/Adam/mul_38training/Adam/mul_39*
_output_shapes
:+*
T0
i
training/Adam/mul_40Multraining/Adam/multraining/Adam/add_22*
_output_shapes
:+*
T0
[
training/Adam/Const_16Const*
_output_shapes
: *
valueB
 *    *
dtype0
[
training/Adam/Const_17Const*
_output_shapes
: *
valueB
 *  �*
dtype0
�
%training/Adam/clip_by_value_8/MinimumMinimumtraining/Adam/add_23training/Adam/Const_17*
_output_shapes
:+*
T0
�
training/Adam/clip_by_value_8Maximum%training/Adam/clip_by_value_8/Minimumtraining/Adam/Const_16*
T0*
_output_shapes
:+
`
training/Adam/Sqrt_8Sqrttraining/Adam/clip_by_value_8*
T0*
_output_shapes
:+
[
training/Adam/add_24/yConst*
_output_shapes
: *
valueB
 *���3*
dtype0
n
training/Adam/add_24Addtraining/Adam/Sqrt_8training/Adam/add_24/y*
_output_shapes
:+*
T0
s
training/Adam/truediv_8RealDivtraining/Adam/mul_40training/Adam/add_24*
_output_shapes
:+*
T0
l
training/Adam/sub_25Subdense_4/bias/readtraining/Adam/truediv_8*
T0*
_output_shapes
:+
�
training/Adam/Assign_21Assigntraining/Adam/Variable_7training/Adam/add_22*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_7*
validate_shape(*
_output_shapes
:+
�
training/Adam/Assign_22Assigntraining/Adam/Variable_15training/Adam/add_23*
T0*,
_class"
 loc:@training/Adam/Variable_15*
validate_shape(*
_output_shapes
:+*
use_locking(
�
training/Adam/Assign_23Assigndense_4/biastraining/Adam/sub_25*
use_locking(*
T0*
_class
loc:@dense_4/bias*
validate_shape(*
_output_shapes
:+
�
training/group_depsNoOp	^loss/add^metrics/acc/Mean^training/Adam/Assign^training/Adam/AssignAdd^training/Adam/Assign_1^training/Adam/Assign_10^training/Adam/Assign_11^training/Adam/Assign_12^training/Adam/Assign_13^training/Adam/Assign_14^training/Adam/Assign_15^training/Adam/Assign_16^training/Adam/Assign_17^training/Adam/Assign_18^training/Adam/Assign_19^training/Adam/Assign_2^training/Adam/Assign_20^training/Adam/Assign_21^training/Adam/Assign_22^training/Adam/Assign_23^training/Adam/Assign_3^training/Adam/Assign_4^training/Adam/Assign_5^training/Adam/Assign_6^training/Adam/Assign_7^training/Adam/Assign_8^training/Adam/Assign_9
0

group_depsNoOp	^loss/add^metrics/acc/Mean
�
IsVariableInitializedIsVariableInitializeddense_1/kernel*!
_class
loc:@dense_1/kernel*
dtype0*
_output_shapes
: 
�
IsVariableInitialized_1IsVariableInitializeddense_1/bias*
_class
loc:@dense_1/bias*
dtype0*
_output_shapes
: 
�
IsVariableInitialized_2IsVariableInitializeddense_2/kernel*!
_class
loc:@dense_2/kernel*
dtype0*
_output_shapes
: 
�
IsVariableInitialized_3IsVariableInitializeddense_2/bias*
_class
loc:@dense_2/bias*
dtype0*
_output_shapes
: 
�
IsVariableInitialized_4IsVariableInitializeddense_3/kernel*!
_class
loc:@dense_3/kernel*
dtype0*
_output_shapes
: 
�
IsVariableInitialized_5IsVariableInitializeddense_3/bias*
_class
loc:@dense_3/bias*
dtype0*
_output_shapes
: 
�
IsVariableInitialized_6IsVariableInitializeddense_4/kernel*!
_class
loc:@dense_4/kernel*
dtype0*
_output_shapes
: 
�
IsVariableInitialized_7IsVariableInitializeddense_4/bias*
_class
loc:@dense_4/bias*
dtype0*
_output_shapes
: 
�
IsVariableInitialized_8IsVariableInitializedAdam/iterations*
_output_shapes
: *"
_class
loc:@Adam/iterations*
dtype0	
z
IsVariableInitialized_9IsVariableInitializedAdam/lr*
_class
loc:@Adam/lr*
dtype0*
_output_shapes
: 
�
IsVariableInitialized_10IsVariableInitializedAdam/beta_1*
_class
loc:@Adam/beta_1*
dtype0*
_output_shapes
: 
�
IsVariableInitialized_11IsVariableInitializedAdam/beta_2*
_class
loc:@Adam/beta_2*
dtype0*
_output_shapes
: 
�
IsVariableInitialized_12IsVariableInitialized
Adam/decay*
_class
loc:@Adam/decay*
dtype0*
_output_shapes
: 
�
IsVariableInitialized_13IsVariableInitializedtraining/Adam/Variable*
dtype0*
_output_shapes
: *)
_class
loc:@training/Adam/Variable
�
IsVariableInitialized_14IsVariableInitializedtraining/Adam/Variable_1*+
_class!
loc:@training/Adam/Variable_1*
dtype0*
_output_shapes
: 
�
IsVariableInitialized_15IsVariableInitializedtraining/Adam/Variable_2*+
_class!
loc:@training/Adam/Variable_2*
dtype0*
_output_shapes
: 
�
IsVariableInitialized_16IsVariableInitializedtraining/Adam/Variable_3*+
_class!
loc:@training/Adam/Variable_3*
dtype0*
_output_shapes
: 
�
IsVariableInitialized_17IsVariableInitializedtraining/Adam/Variable_4*
_output_shapes
: *+
_class!
loc:@training/Adam/Variable_4*
dtype0
�
IsVariableInitialized_18IsVariableInitializedtraining/Adam/Variable_5*
dtype0*
_output_shapes
: *+
_class!
loc:@training/Adam/Variable_5
�
IsVariableInitialized_19IsVariableInitializedtraining/Adam/Variable_6*+
_class!
loc:@training/Adam/Variable_6*
dtype0*
_output_shapes
: 
�
IsVariableInitialized_20IsVariableInitializedtraining/Adam/Variable_7*
dtype0*
_output_shapes
: *+
_class!
loc:@training/Adam/Variable_7
�
IsVariableInitialized_21IsVariableInitializedtraining/Adam/Variable_8*+
_class!
loc:@training/Adam/Variable_8*
dtype0*
_output_shapes
: 
�
IsVariableInitialized_22IsVariableInitializedtraining/Adam/Variable_9*
dtype0*
_output_shapes
: *+
_class!
loc:@training/Adam/Variable_9
�
IsVariableInitialized_23IsVariableInitializedtraining/Adam/Variable_10*,
_class"
 loc:@training/Adam/Variable_10*
dtype0*
_output_shapes
: 
�
IsVariableInitialized_24IsVariableInitializedtraining/Adam/Variable_11*
_output_shapes
: *,
_class"
 loc:@training/Adam/Variable_11*
dtype0
�
IsVariableInitialized_25IsVariableInitializedtraining/Adam/Variable_12*,
_class"
 loc:@training/Adam/Variable_12*
dtype0*
_output_shapes
: 
�
IsVariableInitialized_26IsVariableInitializedtraining/Adam/Variable_13*,
_class"
 loc:@training/Adam/Variable_13*
dtype0*
_output_shapes
: 
�
IsVariableInitialized_27IsVariableInitializedtraining/Adam/Variable_14*,
_class"
 loc:@training/Adam/Variable_14*
dtype0*
_output_shapes
: 
�
IsVariableInitialized_28IsVariableInitializedtraining/Adam/Variable_15*,
_class"
 loc:@training/Adam/Variable_15*
dtype0*
_output_shapes
: 
�
IsVariableInitialized_29IsVariableInitializedtraining/Adam/Variable_16*,
_class"
 loc:@training/Adam/Variable_16*
dtype0*
_output_shapes
: 
�
IsVariableInitialized_30IsVariableInitializedtraining/Adam/Variable_17*,
_class"
 loc:@training/Adam/Variable_17*
dtype0*
_output_shapes
: 
�
IsVariableInitialized_31IsVariableInitializedtraining/Adam/Variable_18*,
_class"
 loc:@training/Adam/Variable_18*
dtype0*
_output_shapes
: 
�
IsVariableInitialized_32IsVariableInitializedtraining/Adam/Variable_19*,
_class"
 loc:@training/Adam/Variable_19*
dtype0*
_output_shapes
: 
�
IsVariableInitialized_33IsVariableInitializedtraining/Adam/Variable_20*,
_class"
 loc:@training/Adam/Variable_20*
dtype0*
_output_shapes
: 
�
IsVariableInitialized_34IsVariableInitializedtraining/Adam/Variable_21*,
_class"
 loc:@training/Adam/Variable_21*
dtype0*
_output_shapes
: 
�
IsVariableInitialized_35IsVariableInitializedtraining/Adam/Variable_22*
dtype0*
_output_shapes
: *,
_class"
 loc:@training/Adam/Variable_22
�
IsVariableInitialized_36IsVariableInitializedtraining/Adam/Variable_23*,
_class"
 loc:@training/Adam/Variable_23*
dtype0*
_output_shapes
: 
�
initNoOp^Adam/beta_1/Assign^Adam/beta_2/Assign^Adam/decay/Assign^Adam/iterations/Assign^Adam/lr/Assign^dense_1/bias/Assign^dense_1/kernel/Assign^dense_2/bias/Assign^dense_2/kernel/Assign^dense_3/bias/Assign^dense_3/kernel/Assign^dense_4/bias/Assign^dense_4/kernel/Assign^training/Adam/Variable/Assign ^training/Adam/Variable_1/Assign!^training/Adam/Variable_10/Assign!^training/Adam/Variable_11/Assign!^training/Adam/Variable_12/Assign!^training/Adam/Variable_13/Assign!^training/Adam/Variable_14/Assign!^training/Adam/Variable_15/Assign!^training/Adam/Variable_16/Assign!^training/Adam/Variable_17/Assign!^training/Adam/Variable_18/Assign!^training/Adam/Variable_19/Assign ^training/Adam/Variable_2/Assign!^training/Adam/Variable_20/Assign!^training/Adam/Variable_21/Assign!^training/Adam/Variable_22/Assign!^training/Adam/Variable_23/Assign ^training/Adam/Variable_3/Assign ^training/Adam/Variable_4/Assign ^training/Adam/Variable_5/Assign ^training/Adam/Variable_6/Assign ^training/Adam/Variable_7/Assign ^training/Adam/Variable_8/Assign ^training/Adam/Variable_9/Assign""� 
trainable_variables� � 
\
dense_1/kernel:0dense_1/kernel/Assigndense_1/kernel/read:02dense_1/random_uniform:08
M
dense_1/bias:0dense_1/bias/Assigndense_1/bias/read:02dense_1/Const:08
\
dense_2/kernel:0dense_2/kernel/Assigndense_2/kernel/read:02dense_2/random_uniform:08
M
dense_2/bias:0dense_2/bias/Assigndense_2/bias/read:02dense_2/Const:08
\
dense_3/kernel:0dense_3/kernel/Assigndense_3/kernel/read:02dense_3/random_uniform:08
M
dense_3/bias:0dense_3/bias/Assigndense_3/bias/read:02dense_3/Const:08
\
dense_4/kernel:0dense_4/kernel/Assigndense_4/kernel/read:02dense_4/random_uniform:08
M
dense_4/bias:0dense_4/bias/Assigndense_4/bias/read:02dense_4/Const:08
f
Adam/iterations:0Adam/iterations/AssignAdam/iterations/read:02Adam/iterations/initial_value:08
F
	Adam/lr:0Adam/lr/AssignAdam/lr/read:02Adam/lr/initial_value:08
V
Adam/beta_1:0Adam/beta_1/AssignAdam/beta_1/read:02Adam/beta_1/initial_value:08
V
Adam/beta_2:0Adam/beta_2/AssignAdam/beta_2/read:02Adam/beta_2/initial_value:08
R
Adam/decay:0Adam/decay/AssignAdam/decay/read:02Adam/decay/initial_value:08
q
training/Adam/Variable:0training/Adam/Variable/Assigntraining/Adam/Variable/read:02training/Adam/zeros:08
y
training/Adam/Variable_1:0training/Adam/Variable_1/Assigntraining/Adam/Variable_1/read:02training/Adam/zeros_1:08
y
training/Adam/Variable_2:0training/Adam/Variable_2/Assigntraining/Adam/Variable_2/read:02training/Adam/zeros_2:08
y
training/Adam/Variable_3:0training/Adam/Variable_3/Assigntraining/Adam/Variable_3/read:02training/Adam/zeros_3:08
y
training/Adam/Variable_4:0training/Adam/Variable_4/Assigntraining/Adam/Variable_4/read:02training/Adam/zeros_4:08
y
training/Adam/Variable_5:0training/Adam/Variable_5/Assigntraining/Adam/Variable_5/read:02training/Adam/zeros_5:08
y
training/Adam/Variable_6:0training/Adam/Variable_6/Assigntraining/Adam/Variable_6/read:02training/Adam/zeros_6:08
y
training/Adam/Variable_7:0training/Adam/Variable_7/Assigntraining/Adam/Variable_7/read:02training/Adam/zeros_7:08
y
training/Adam/Variable_8:0training/Adam/Variable_8/Assigntraining/Adam/Variable_8/read:02training/Adam/zeros_8:08
y
training/Adam/Variable_9:0training/Adam/Variable_9/Assigntraining/Adam/Variable_9/read:02training/Adam/zeros_9:08
}
training/Adam/Variable_10:0 training/Adam/Variable_10/Assign training/Adam/Variable_10/read:02training/Adam/zeros_10:08
}
training/Adam/Variable_11:0 training/Adam/Variable_11/Assign training/Adam/Variable_11/read:02training/Adam/zeros_11:08
}
training/Adam/Variable_12:0 training/Adam/Variable_12/Assign training/Adam/Variable_12/read:02training/Adam/zeros_12:08
}
training/Adam/Variable_13:0 training/Adam/Variable_13/Assign training/Adam/Variable_13/read:02training/Adam/zeros_13:08
}
training/Adam/Variable_14:0 training/Adam/Variable_14/Assign training/Adam/Variable_14/read:02training/Adam/zeros_14:08
}
training/Adam/Variable_15:0 training/Adam/Variable_15/Assign training/Adam/Variable_15/read:02training/Adam/zeros_15:08
}
training/Adam/Variable_16:0 training/Adam/Variable_16/Assign training/Adam/Variable_16/read:02training/Adam/zeros_16:08
}
training/Adam/Variable_17:0 training/Adam/Variable_17/Assign training/Adam/Variable_17/read:02training/Adam/zeros_17:08
}
training/Adam/Variable_18:0 training/Adam/Variable_18/Assign training/Adam/Variable_18/read:02training/Adam/zeros_18:08
}
training/Adam/Variable_19:0 training/Adam/Variable_19/Assign training/Adam/Variable_19/read:02training/Adam/zeros_19:08
}
training/Adam/Variable_20:0 training/Adam/Variable_20/Assign training/Adam/Variable_20/read:02training/Adam/zeros_20:08
}
training/Adam/Variable_21:0 training/Adam/Variable_21/Assign training/Adam/Variable_21/read:02training/Adam/zeros_21:08
}
training/Adam/Variable_22:0 training/Adam/Variable_22/Assign training/Adam/Variable_22/read:02training/Adam/zeros_22:08
}
training/Adam/Variable_23:0 training/Adam/Variable_23/Assign training/Adam/Variable_23/read:02training/Adam/zeros_23:08"� 
	variables� � 
\
dense_1/kernel:0dense_1/kernel/Assigndense_1/kernel/read:02dense_1/random_uniform:08
M
dense_1/bias:0dense_1/bias/Assigndense_1/bias/read:02dense_1/Const:08
\
dense_2/kernel:0dense_2/kernel/Assigndense_2/kernel/read:02dense_2/random_uniform:08
M
dense_2/bias:0dense_2/bias/Assigndense_2/bias/read:02dense_2/Const:08
\
dense_3/kernel:0dense_3/kernel/Assigndense_3/kernel/read:02dense_3/random_uniform:08
M
dense_3/bias:0dense_3/bias/Assigndense_3/bias/read:02dense_3/Const:08
\
dense_4/kernel:0dense_4/kernel/Assigndense_4/kernel/read:02dense_4/random_uniform:08
M
dense_4/bias:0dense_4/bias/Assigndense_4/bias/read:02dense_4/Const:08
f
Adam/iterations:0Adam/iterations/AssignAdam/iterations/read:02Adam/iterations/initial_value:08
F
	Adam/lr:0Adam/lr/AssignAdam/lr/read:02Adam/lr/initial_value:08
V
Adam/beta_1:0Adam/beta_1/AssignAdam/beta_1/read:02Adam/beta_1/initial_value:08
V
Adam/beta_2:0Adam/beta_2/AssignAdam/beta_2/read:02Adam/beta_2/initial_value:08
R
Adam/decay:0Adam/decay/AssignAdam/decay/read:02Adam/decay/initial_value:08
q
training/Adam/Variable:0training/Adam/Variable/Assigntraining/Adam/Variable/read:02training/Adam/zeros:08
y
training/Adam/Variable_1:0training/Adam/Variable_1/Assigntraining/Adam/Variable_1/read:02training/Adam/zeros_1:08
y
training/Adam/Variable_2:0training/Adam/Variable_2/Assigntraining/Adam/Variable_2/read:02training/Adam/zeros_2:08
y
training/Adam/Variable_3:0training/Adam/Variable_3/Assigntraining/Adam/Variable_3/read:02training/Adam/zeros_3:08
y
training/Adam/Variable_4:0training/Adam/Variable_4/Assigntraining/Adam/Variable_4/read:02training/Adam/zeros_4:08
y
training/Adam/Variable_5:0training/Adam/Variable_5/Assigntraining/Adam/Variable_5/read:02training/Adam/zeros_5:08
y
training/Adam/Variable_6:0training/Adam/Variable_6/Assigntraining/Adam/Variable_6/read:02training/Adam/zeros_6:08
y
training/Adam/Variable_7:0training/Adam/Variable_7/Assigntraining/Adam/Variable_7/read:02training/Adam/zeros_7:08
y
training/Adam/Variable_8:0training/Adam/Variable_8/Assigntraining/Adam/Variable_8/read:02training/Adam/zeros_8:08
y
training/Adam/Variable_9:0training/Adam/Variable_9/Assigntraining/Adam/Variable_9/read:02training/Adam/zeros_9:08
}
training/Adam/Variable_10:0 training/Adam/Variable_10/Assign training/Adam/Variable_10/read:02training/Adam/zeros_10:08
}
training/Adam/Variable_11:0 training/Adam/Variable_11/Assign training/Adam/Variable_11/read:02training/Adam/zeros_11:08
}
training/Adam/Variable_12:0 training/Adam/Variable_12/Assign training/Adam/Variable_12/read:02training/Adam/zeros_12:08
}
training/Adam/Variable_13:0 training/Adam/Variable_13/Assign training/Adam/Variable_13/read:02training/Adam/zeros_13:08
}
training/Adam/Variable_14:0 training/Adam/Variable_14/Assign training/Adam/Variable_14/read:02training/Adam/zeros_14:08
}
training/Adam/Variable_15:0 training/Adam/Variable_15/Assign training/Adam/Variable_15/read:02training/Adam/zeros_15:08
}
training/Adam/Variable_16:0 training/Adam/Variable_16/Assign training/Adam/Variable_16/read:02training/Adam/zeros_16:08
}
training/Adam/Variable_17:0 training/Adam/Variable_17/Assign training/Adam/Variable_17/read:02training/Adam/zeros_17:08
}
training/Adam/Variable_18:0 training/Adam/Variable_18/Assign training/Adam/Variable_18/read:02training/Adam/zeros_18:08
}
training/Adam/Variable_19:0 training/Adam/Variable_19/Assign training/Adam/Variable_19/read:02training/Adam/zeros_19:08
}
training/Adam/Variable_20:0 training/Adam/Variable_20/Assign training/Adam/Variable_20/read:02training/Adam/zeros_20:08
}
training/Adam/Variable_21:0 training/Adam/Variable_21/Assign training/Adam/Variable_21/read:02training/Adam/zeros_21:08
}
training/Adam/Variable_22:0 training/Adam/Variable_22/Assign training/Adam/Variable_22/read:02training/Adam/zeros_22:08
}
training/Adam/Variable_23:0 training/Adam/Variable_23/Assign training/Adam/Variable_23/read:02training/Adam/zeros_23:08�N�       ���	�a[�B=�A*

val_loss�^�?C��J       �	xc[�B=�A*

val_acc])�=��       �K"	5d[�B=�A*

loss��@5r       ���	�d[�B=�A*


accjyv=.�p       ��2	ec�B=�A*

val_loss�u�?�%��       `/�#	Zfc�B=�A*

val_acc
h=�!�       ��-	�fc�B=�A*

loss�'�?)��d       ��(	Hgc�B=�A*


accW˄=�=�I       ��2	�	k�B=�A*

val_lossP�?p)��       `/�#	�
k�B=�A*

val_acc�"�=�Zb       ��-	�k�B=�A*

loss:�?HH�	       ��(	�k�B=�A*


acc��|=��4       ��2	��r�B=�A*

val_lossi_�?r �s       `/�#	\�r�B=�A*

val_accbd�=U�G(       ��-	�r�B=�A*

loss!�?5}�       ��(	��r�B=�A*


accG�=�WZ       ��2	<�z�B=�A*

val_lossܒ�?��Pc       `/�#	��z�B=�A*

val_accbd�=�2\�       ��-	��z�B=�A*

loss$9�?c�)       ��(	�z�B=�A*


accvl�=��Q]       ��2	�c��B=�A*

val_loss0i?�	�       `/�#	ke��B=�A*

val_acc�=�=%#jA       ��-	(f��B=�A*

loss���?�ܻ�       ��(	�f��B=�A*


acc\��=���       ��2	xc��B=�A*

val_loss1[u?t�(       `/�#	e��B=�A*

val_acc��W=N�x       ��-	�e��B=�A*

loss�$�?�r�e       ��(	{f��B=�A*


acc%\c=����       ��2	H2��B=�A*

val_loss�o?e�z�       `/�#	j3��B=�A*

val_acc��==Tk܉       ��-	�3��B=�A*

loss(Az?[�       ��(	n4��B=�A*


acc43=����       ��2	�%��B=�A*

val_loss4�i?��D�       `/�#	N'��B=�A*

val_acc��/=�`{       ��-	(��B=�A*

loss2gt?�#C�       ��(	�(��B=�A*


acc�1=܌}�       ��2	����B=�A	*

val_lossV&e?)O�       `/�#	����B=�A	*

val_acc�f=�>�g       ��-	8���B=�A	*

loss�Po?,�q�       ��(	����B=�A	*


acc�
.=�Fe       ��2	4.��B=�A
*

val_loss�`?UD��       `/�#	�/��B=�A
*

val_acc&��<���J       ��-	�0��B=�A
*

losscj?-ʗ�       ��(	&1��B=�A
*


acc�	={��.       ��2	�L��B=�A*

val_loss�~\?���b       `/�#	�N��B=�A*

val_accFt%=k       ��-	jO��B=�A*

loss��e?s�5G       ��(	P��B=�A*


acc�^=��fj       ��2	>���B=�A*

val_loss�Y?N���       `/�#	�B=�A*

val_acc��==B�uH       ��-	AﺔB=�A*

loss�a?�%N       ��(	�ﺔB=�A*


acc4I=|\r       ��2	<2ÔB=�A*

val_loss�4V?F�       `/�#	�3ÔB=�A*

val_acc�M=+ݑ       ��-	�4ÔB=�A*

loss�V^?7p޸       ��(	75ÔB=�A*


accn�W=*���       ��2	�˔B=�A*

val_loss%�S?A�l�       `/�#	�˔B=�A*

val_acc��K=���2       ��-	E˔B=�A*

lossЌ[?���       ��(	�˔B=�A*


acc��]=� �       ��2	6�ҔB=�A*

val_loss%uQ?�%�       `/�#	��ҔB=�A*

val_acc&�Q=B���       ��-	��ҔB=�A*

loss(	Y?��       ��(	)�ҔB=�A*


accqT^=Y�       ��2	��ڔB=�A*

val_loss��O?�~2       `/�#	S�ڔB=�A*

val_acc&�Q=��%       ��-	�ڔB=�A*

loss�W?���       ��(	��ڔB=�A*


accqT^=?��H       ��2	|C�B=�A*

val_loss'�N?.��       `/�#	E�B=�A*

val_acc&�Q=�R��       ��-	�E�B=�A*

lossH]U?Mv�       ��(	wF�B=�A*


accqT^=�Љ       ��2	z��B=�A*

val_loss
*M?���       `/�#	��B=�A*

val_acc&�Q=��ݧ       ��-	���B=�A*

loss��S?SS��       ��(	u��B=�A*


accqT^=���       ��2	���B=�A*

val_loss�ZL?8:�        `/�#	;��B=�A*

val_acc&�Q=���       ��-	 ��B=�A*

loss;�R?�7˜       ��(	���B=�A*


accqT^=j�k       ��2	���B=�A*

val_losslK?��2       `/�#	{��B=�A*

val_acc��S=:�N&       ��-	���B=�A*

loss�Q?���       ��(	���B=�A*


accqT^=�G;       ��2	�q�B=�A*

val_losstUJ?��{8       `/�#	~s�B=�A*

val_acc��S=L��g       ��-	Gt�B=�A*

loss �P?l��        ��(	�t�B=�A*


acc�U_=*?�       ��2	���B=�A*

val_lossi�I?!���       `/�#	U��B=�A*

val_acc&�Q=8���       ��-	��B=�A*

lossҚO?$
ĉ       ��(	���B=�A*


acc�U_=΢;x       ��2	�B=�A*

val_loss��H?�T�       `/�#	��B=�A*

val_acc��S=�60c       ��-	a�B=�A*

loss��N?�xD�       ��(	 �B=�A*


acc��_=�-R       ��2	3R�B=�A*

val_loss�UH?�M|       `/�#	�S�B=�A*

val_acc��S=���       ��-	�T�B=�A*

loss2�M?%�!W       ��(	%U�B=�A*


acc�U_=�e�       ��2	��&�B=�A*

val_lossK�G?��ƕ       `/�#	#�&�B=�A*

val_acc&�Q=�~       ��-	��&�B=�A*

loss�,M?J��@       ��(	z�&�B=�A*


acc�U_=X�*�       ��2	�A/�B=�A*

val_lossUG?�5�       `/�#	RC/�B=�A*

val_acc;�U=�S}       ��-	D/�B=�A*

lossf_L?o�A       ��(	�D/�B=�A*


acc�U_=?�r       ��2	�F7�B=�A*

val_loss|!F?}h��       `/�#	,H7�B=�A*

val_acc&�Q=ŧ�]       ��-	�H7�B=�A*

lossGeK?�bT�       ��(	�I7�B=�A*


acc�U_=��p�       ��2	ާ@�B=�A*

val_loss޳E?���       `/�#	�@�B=�A*

val_acc��O=���.       ��-	��@�B=�A*

loss��J?��t       ��(	�@�B=�A*


acc�W`=�1�       ��2	��H�B=�A*

val_lossy�D?|Vǚ       `/�#	��H�B=�A*

val_acc��W=�\A       ��-	E�H�B=�A*

loss��I?����       ��(	��H�B=�A*


acc��a=�N��       ��2	6Q�B=�A*

val_lossmD?)4"       `/�#	�Q�B=�A*

val_acc�M=eau&       ��-	�Q�B=�A*

loss�H?�Q       ��(	$	Q�B=�A*


acc�Z=U�>r       ��2	�X�B=�A*

val_loss?^C?Y=�       `/�#	^�X�B=�A*

val_acc_�C=A��       ��-	��X�B=�A*

loss��G?v��z       ��(	8�X�B=�A*


acc�O=�Ӻ�       ��2	�b�B=�A *

val_loss1�B?|S       `/�#	��b�B=�A *

val_accY=����       ��-	��b�B=�A *

lossk�F?�"�       ��(	��b�B=�A *


acc�>=K 	*       ��2	�&j�B=�A!*

val_lossu�A?�ø�       `/�#	�'j�B=�A!*

val_accUB=/�)5       ��-	[(j�B=�A!*

loss��E??�+�       ��(	�(j�B=�A!*


accW*=��9]       ��2	`rq�B=�A"*

val_loss+9A?�%g�       `/�#	�sq�B=�A"*

val_acc��5=a-�M       ��-	�tq�B=�A"*

loss ;E?�᫄       ��(	Suq�B=�A"*


acc?w!=,�Ї       ��2	��x�B=�A#*

val_losswY@?v���       `/�#	&�x�B=�A#*

val_acco�-=FUhs       ��-	��x�B=�A#*

loss��D?tC�/       ��(	��x�B=�A#*


acc*t= ��       ��2	9��B=�A$*

val_loss �??)/��       `/�#	��B=�A$*

val_acc��/=����       ��-	���B=�A$*

loss��C?�x/�       ��(	���B=�A$*


acc'�=,w��       ��2	x���B=�A%*

val_loss�7??��c�       `/�#	���B=�A%*

val_accjK=�e��       ��-	̵��B=�A%*

lossC?���       ��(	k���B=�A%*


acc*t=1��s       ��2	U��B=�A&*

val_loss�x>?(4��       `/�#	���B=�A&*

val_accA9=�C��       ��-	���B=�A&*

loss�FB?Q�o�       ��(	C��B=�A&*


accy� =#
�       ��2	���B=�A'*

val_lossW[>?�2�*       `/�#	}���B=�A'*

val_acco�-=���f       ��-	C���B=�A'*

loss|�A?'���       ��(	ޯ��B=�A'*


acch}%=�'q�       ��2	I��B=�A(*

val_loss��=?�D�       `/�#	���B=�A(*

val_acc,0=a�{"       ��-	���B=�A(*

loss1rA?�z�       ��(	D��B=�A(*


acc��$=|�؋       ��2	u��B=�A)*

val_loss�F=?�X�       `/�#	�v��B=�A)*

val_acc6�;= wv5       ��-	)w��B=�A)*

loss�m@?��C�       ��(	�w��B=�A)*


acc�0=��K       ��2	����B=�A**

val_loss�=?�zH�       `/�#	K���B=�A**

val_accd=�938       ��-	���B=�A**

loss��??���l       ��(	����B=�A**


acc�U_=$�E       ��2	���B=�A+*

val_loss1�<?z�       `/�#	x��B=�A+*

val_acch.x=z��9       ��-	<��B=�A+*

loss%�?? ��       ��(	���B=�A+*


accD�x=f��       ��2	���B=�A,*

val_lossCf<?I,�       `/�#	����B=�A,*

val_acc@�=˯��       ��-	u���B=�A,*

loss�l??
��v       ��(	���B=�A,*


acc�{=�-h�       ��2	��ĕB=�A-*

val_loss�&<?�F�"       `/�#	�ĕB=�A-*

val_acc�h�=N�fg       ��-	_�ĕB=�A-*

lossT�>?�t��       ��(	̘ĕB=�A-*


accҘ�=�=       ��2	�(̕B=�A.*

val_lossл;?�zl�       `/�#	�+̕B=�A.*

val_acc�=�=f��       ��-	�-̕B=�A.*

loss&�>?�"�       ��(	�.̕B=�A.*


accҘ�=�|�       ��2	ҎӕB=�A/*

val_loss�{;?��
       `/�#	��ӕB=�A/*

val_acc�o�=��       ��-	i�ӕB=�A/*

lossc�=?��w       ��(	O�ӕB=�A/*


acct'�=�KjY       ��2	�ەB=�A0*

val_loss�D;?sG       `/�#	�ەB=�A0*

val_acc���=�g.�       ��-	3�ەB=�A0*

lossY�=?���       ��(	&�ەB=�A0*


accai�=����       ��2	i��B=�A1*

val_lossu�:?iLLa       `/�#	v��B=�A1*

val_acc)��=�.\%       ��-	���B=�A1*

loss�@=?��,       ��(	7��B=�A1*


acc���=yd,       ��2	���B=�A2*

val_lossJ%;?����       `/�#	���B=�A2*

val_acc���=
���       ��-	���B=�A2*

lossL�<?�kws       ��(	���B=�A2*


acc���=,=0�       ��2	E��B=�A3*

val_loss-\;?Q8��       `/�#	���B=�A3*

val_acc~T�=�X�       ��-	E��B=�A3*

loss��<?����       ��(	���B=�A3*


acc,��=/�       ��2	>��B=�A4*

val_loss�O:?@c       `/�#	���B=�A4*

val_acc���=���       ��-	`��B=�A4*

loss{<?G��       ��(	���B=�A4*


accB�=jQ~[       ��2	�q��B=�A5*

val_lossN:?�7]�       `/�#	�r��B=�A5*

val_acc���=�=�       ��-	Cs��B=�A5*

loss�H<?oˈy       ��(	�s��B=�A5*


acc���=織�       ��2	���B=�A6*

val_loss��9?a��Y       `/�#	���B=�A6*

val_acc��=?���       ��-	P��B=�A6*

lossR�;?�U�       ��(	���B=�A6*


acc-��=�IQ9       ��2	X�
�B=�A7*

val_loss��:?�ul:       `/�#	F�
�B=�A7*

val_acc���=T\�       ��-	��
�B=�A7*

lossר;?< �m       ��(	1�
�B=�A7*


acc
�=c��       ��2	AJ�B=�A8*

val_loss�-:?P탏       `/�#	�K�B=�A8*

val_accC��=��D�       ��-	^L�B=�A8*

lossN{;?�'2       ��(	'M�B=�A8*


acc�ܹ=��A       ��2	-��B=�A9*

val_loss�(9??��       `/�#	g��B=�A9*

val_acc��=�)�H       ��-	���B=�A9*

lossk%;?q��       ��(	���B=�A9*


acc&#�=8ūE       ��2	�^�B=�A:*

val_loss*J9?���-       `/�#	�_�B=�A:*

val_acc���=k/_`       ��-	`�B=�A:*

loss{�:?����       ��(	�`�B=�A:*


acc"��=r#D       ��2	@%�B=�A;*

val_lossk�8?
�?c       `/�#	�%�B=�A;*

val_acc���=�s��       ��-	�%�B=�A;*

loss�:?g'�       ��(	3%�B=�A;*


acc@��=M��       ��2	�G*�B=�A<*

val_loss��8?���       `/�#	�H*�B=�A<*

val_accG�=P$��       ��-	=I*�B=�A<*

loss(e:?(�a<       ��(	�I*�B=�A<*


accT��=�Ü       ��2	z/�B=�A=*

val_lossW�8?�}"�       `/�#	x{/�B=�A=*

val_acc+�=Hc:�       ��-	�{/�B=�A=*

lossU�:?���       ��(	_|/�B=�A=*


acc�I�=��r0       ��2	�f7�B=�A>*

val_loss��7?�R�)       `/�#	(h7�B=�A>*

val_acc}7�=�St       ��-	�h7�B=�A>*

loss��9?���       ��(	i7�B=�A>*


accOb�=�"�i       ��2	p_<�B=�A?*

val_lossYM8?ⴖ�       `/�#	�`<�B=�A?*

val_acc��>��F�       ��-	�a<�B=�A?*

loss��9?���
       ��(	Ab<�B=�A?*


acct�>"�hv       ��2	.D�B=�A@*

val_loss~�7?���       `/�#	�/D�B=�A@*

val_acc}7�=C��       ��-	^0D�B=�A@*

lossb�9? 9�g       ��(	�0D�B=�A@*


accf�>a��       ��2	�rI�B=�AA*

val_loss�8?�C�u       `/�#	tI�B=�AA*

val_acc�;>�4n       ��-	�tI�B=�AA*

loss�"9?��Ov       ��(	uI�B=�AA*


accĩ>�L�)       ��2	��P�B=�AB*

val_loss^{7? A�       `/�#	<�P�B=�AB*

val_acc�]>���       ��-	��P�B=�AB*

loss{�8?�m�       ��(	��P�B=�AB*


acc��!>��)       ��2	�`X�B=�AC*

val_loss�k7?'!#       `/�#	�bX�B=�AC*

val_acc��1>Q��       ��-	JcX�B=�AC*

loss�{8?����       ��(	�cX�B=�AC*


acc�d*>J�       ��2	�S]�B=�AD*

val_loss'�7?�z�       `/�#	�T]�B=�AD*

val_acc��5>K��       ��-	aU]�B=�AD*

lossiX8?,���       ��(	�U]�B=�AD*


acc�d*>�5�M       ��2	�/e�B=�AE*

val_loss?�6?-�E�       `/�#	f1e�B=�AE*

val_acc�:>�v��       ��-	#2e�B=�AE*

lossH8?�;�       ��(	�2e�B=�AE*


acc�02>s�ē       ��2	?�j�B=�AF*

val_loss��6?u�       `/�#	.�j�B=�AF*

val_acc��I>��^        ��-	��j�B=�AF*

loss��7?`۷0       ��(	.�j�B=�AF*


acc�8>&��L       ��2	&r�B=�AG*

val_loss�k6?'�I       `/�#	�'r�B=�AG*

val_acc�B>����       ��-	h(r�B=�AG*

loss&�7?)Ӑ       ��(	)r�B=�AG*


acc�8>��D       ��2	�2w�B=�AH*

val_loss��6?�IY^       `/�#	�3w�B=�AH*

val_acc��L>�)X�       ��-	n4w�B=�AH*

lossgs7?C&	       ��(	�4w�B=�AH*


acc.=:>���       ��2	�;|�B=�AI*

val_loss��6?V��L       `/�#	y=|�B=�AI*

val_acc{�<>�4�       ��-	>|�B=�AI*

loss�K7?��9�       ��(	�>|�B=�AI*


acc�>;>�4�c       ��2	�u��B=�AJ*

val_loss/�6?��ZK       `/�#	�v��B=�AJ*

val_acc�2A>��       ��-	`w��B=�AJ*

loss	�7?{kA�       ��(	�w��B=�AJ*


accr9>уL|       ��2	���B=�AK*

val_loss��5?be�       `/�#	���B=�AK*

val_acc��K>y˨p       ��-	
��B=�AK*

loss�m7?��vJ       ��(	��B=�AK*


acca�=>�0x�       ��2	ۡ��B=�AL*

val_loss�5? x'�       `/�#	
���B=�AL*

val_acc�;E>zL��       ��-	����B=�AL*

loss��6?��       ��(	����B=�AL*


acc�GA>�Q�       ��2	�엖B=�AM*

val_loss|5?7u�       `/�#	�B=�AM*

val_acc�KL>�C
�       ��-	�B=�AM*

lossۅ6?�<d       ��(	��B=�AM*


acc	B>��       ��2	�윖B=�AN*

val_loss}�5?�*       `/�#	|휖B=�AN*

val_acc��S>�ɓ�       ��-	�휖B=�AN*

loss��6?���       ��(	FB=�AN*


acc/�=>��       ��2	⤖B=�AO*

val_lossyR5?q��       `/�#	C㤖B=�AO*

val_acc�;E>�ne.       ��-	�㤖B=�AO*

loss�.6?V&P       ��(	2䤖B=�AO*


acc�C>��E�       ��2	���B=�AP*

val_loss�]5?��؏       `/�#	���B=�AP*

val_accWQ>��S       ��-	b��B=�AP*

lossH6?�J       ��(	���B=�AP*


acc�A>6��       ��2	����B=�AQ*

val_loss
5?v�x^       `/�#	 ���B=�AQ*

val_acc.�F>��$
       ��-	)���B=�AQ*

loss� 6?�|	       ��(	돱�B=�AQ*


acc��A>�r�5       ��2	ն�B=�AR*

val_loss2H5?[f��       `/�#	�ն�B=�AR*

val_accs�G>j��       ��-	wֶ�B=�AR*

loss{�5?��#k       ��(	�ֶ�B=�AR*


acc��D>]R�P       ��2	D��B=�AS*

val_loss�4?9e�a       `/�#	�E��B=�AS*

val_accW�N>|�o`       ��-	cF��B=�AS*

loss��5?�or       ��(	�F��B=�AS*


acc�D?>b1�I       ��2	�tÖB=�AT*

val_loss�4?X��       `/�#	�uÖB=�AT*

val_accC�J>.���       ��-	CvÖB=�AT*

loss<�5?4y       ��(	�vÖB=�AT*


acc�E>���       ��2	t�ɖB=�AU*

val_loss^5?c��       `/�#	��ɖB=�AU*

val_accs�G>�&�3       ��-	��ɖB=�AU*

loss��5?���       ��(	0�ɖB=�AU*


acc�iB>ԝ�y       ��2	�ҖB=�AV*

val_loss��4?M-:�       `/�#	ҖB=�AV*

val_acc�+>>g�hj       ��-	��ҖB=�AV*

lossES5? is       ��(	�ҖB=�AV*


accn�C>�Q�&       ��2	�ۖB=�AW*

val_lossM4?��$W       `/�#	DۖB=�AW*

val_acc.�F>i��N       ��-	VJۖB=�AW*

loss�5?�i\       ��(	/LۖB=�AW*


acc3D>SGY�       ��2	���B=�AX*

val_loss]4?����       `/�#	~��B=�AX*

val_acc��K>�c�       ��-	\��B=�AX*

loss:�4?n��r       ��(	&��B=�AX*


acc�E>��       ��2	��B=�AY*

val_loss\�4?�ژ       `/�#	��B=�AY*

val_acc� 9>�7L       ��-	��B=�AY*

loss=�4?tO�6       ��(	Y�B=�AY*


accd,D>�9       ��2	M��B=�AZ*

val_loss'�4?ɋ�       `/�#	<��B=�AZ*

val_acc�BH>��d       ��-	���B=�AZ*

loss�!5?�y�9       ��(	8��B=�AZ*


acc��A>|
`[       ��2	�>��B=�A[*

val_lossa4?]'�b       `/�#	)@��B=�A[*

val_acc�DI>�V|       ��-	�@��B=�A[*

loss~�4?��x
       ��(	A��B=�A[*


acc�iB>����       ��2	���B=�A\*

val_lossrl4?,*�       `/�#	���B=�A\*

val_acc�6>=p��       ��-	.��B=�A\*

loss�4?��d       ��(	���B=�A\*


acc�B>ċHR       ��2	���B=�A]*

val_loss^�3?^�W*       `/�#	7��B=�A]*

val_acc�[S>-�"       ��-	"��B=�A]*

loss6�4?��       ��(	���B=�A]*


accy/F>���       ��2	v6�B=�A^*

val_loss#z4?w_       `/�#	H8�B=�A^*

val_acc�2A>Y�`       ��-	�8�B=�A^*

lossM4?YPZ�       ��(	*9�B=�A^*


acc��G>r�X       ��2	Q1"�B=�A_*

val_loss��4?(!�]       `/�#	 3"�B=�A_*

val_acc�MM>vy�R       ��-	�3"�B=�A_*

loss{�4?1�/�       ��(	�4"�B=�A_*


acc�*C>~�       ��2	J@-�B=�A`*

val_loss�y3?I?&�       `/�#	�A-�B=�A`*

val_acc&�Q>t"�%       ��-	(B-�B=�A`*

loss�>4?       ��(	�B-�B=�A`*


acc�IB>8�       ��2	&9�B=�Aa*

val_loss�i3?���       `/�#	9�B=�Aa*

val_acc<7C>�t��       ��-	�9�B=�Aa*

lossm4?:�:       ��(	�9�B=�Aa*


acc��E>!p�:       ��2	l�C�B=�Ab*

val_loss�3?���       `/�#	!�C�B=�Ab*

val_acceIK>W��       ��-	�C�B=�Ab*

loss��3?˸��       ��(	��C�B=�Ab*


acc��A>x{Z       ��2	`�J�B=�Ac*

val_loss�|3?9�i�       `/�#	P�J�B=�Ac*

val_acc��D>��w       ��-	��J�B=�Ac*

loss3?�G       ��(	�J�B=�Ac*


accQnE>��S       ��2	�tT�B=�Ad*

val_loss�:3?�-�       `/�#	.vT�B=�Ad*

val_accm0@>����       ��-	�vT�B=�Ad*

lossן3?\�       ��(	�wT�B=�Ad*


acc?>����       ��2	[`�B=�Ae*

val_loss�G3?�`-       `/�#	k`�B=�Ae*

val_acc.�F>-���       ��-	^`�B=�Ae*

loss�b3?�?i       ��(	(`�B=�Ae*


acc NE>� �       ��2	;�i�B=�Af*

val_lossT�2?��       `/�#	L�i�B=�Af*

val_acc�E>����       ��-	O�i�B=�Af*

loss�I3?v�D       ��(	��i�B=�Af*


accO)B>�84       ��2	��u�B=�Ag*

val_loss?�2?�!��       `/�#	C�u�B=�Ag*

val_accQ@G>�S�       ��-	%�u�B=�Ag*

loss��2?Isn       ��(	��u�B=�Ag*


acc�-E>��T       ��2	q��B=�Ah*

val_loss�z2?�3       `/�#	W��B=�Ah*

val_acc�BH>
��       ��-	̴�B=�Ah*

loss{3?���       ��(	���B=�Ah*


acc��E>�7�n       ��2	I~��B=�Ai*

val_loss�h2?���       `/�#	����B=�Ai*

val_acc��L>��[�       ��-	����B=�Ai*

loss��2?����       ��(	I���B=�Ai*


acc1�@>�vi       ��2	�B��B=�Aj*

val_loss1`2?!�       `/�#	OD��B=�Aj*

val_acc<7C>Y�L       ��-	E��B=�Aj*

loss��2?�uR�       ��(	�E��B=�Aj*


acc�B>���       ��2	�E��B=�Ak*

val_loss�1?�
��       `/�#	�K��B=�Ak*

val_acc��K>�&��       ��-	�M��B=�Ak*

loss&2?1�L       ��(	O��B=�Ak*


acc��D>;j_       ��2	�ո�B=�Al*

val_loss$�1?��V�       `/�#	Uٸ�B=�Al*

val_acczRO>T��       ��-	�ڸ�B=�Al*

loss�1?_$�B       ��(	�ݸ�B=�Al*


accZ�D>�U �       ��2	�x��B=�Am*

val_loss�\2?�E3       `/�#	z��B=�Am*

val_accIYR>{k�       ��-	�z��B=�Am*

loss�02?����       ��(	�z��B=�Am*


acc�oF>C�e       ��2	G˗B=�An*

val_lossY�2?=���       `/�#	MH˗B=�An*

val_acc4PN>v.*�       ��-	I˗B=�An*

loss��2?�R��       ��(	�I˗B=�An*


accw�B>E�,       ��2	LQԗB=�Ao*

val_loss��1?����       `/�#	�RԗB=�Ao*

val_acc�DI>`љ       ��-	�SԗB=�Ao*

lossU�1?��g�       ��(	+TԗB=�Ao*


acc�lD>V�o�       ��2	P�ܗB=�Ap*

val_loss(�1?�L��       `/�#	��ܗB=�Ap*

val_acc`U>��[�       ��-	r�ܗB=�Ap*

losse�1?�̂�       ��(	�ܗB=�Ap*


acco�F>����       ��2	��B=�Aq*

val_loss��1?m&Ҕ       `/�#	��B=�Aq*

val_acc�DI>����       ��-	Ơ�B=�Aq*

loss7�1?����       ��(	b��B=�Aq*


acc��E>�͍�       ��2	\��B=�Ar*

val_loss:J1?��       `/�#	2�B=�Ar*

val_acc�KL>kݏ3       ��-	��B=�Ar*

lossT�1?��&       ��(	;�B=�Ar*


acc1G>S��       ��2	f���B=�As*

val_lossn1?�	
�       `/�#	����B=�As*

val_acc
�X>�X�-       ��-	7���B=�As*

loss�C1?VBI"       ��(	ŋ��B=�As*


acc��H>�°�       ��2	�4	�B=�At*

val_loss�1?�;J�       `/�#	i6	�B=�At*

val_accC�J>}�̺       ��-	*7	�B=�At*

loss�A1?�>�O       ��(	�7	�B=�At*


accvJ>A���       ��2	��B=�Au*

val_loss1?Ƭ^       `/�#	!�B=�Au*

val_accrkZ>���       ��-	��B=�Au*

loss,1?�6�       ��(	d�B=�Au*


acc^WK>=��^       ��2	:A%�B=�Av*

val_loss�0?2�j�       `/�#	uB%�B=�Av*

val_acc�}b>��`       ��-	�B%�B=�Av*

loss��0?u�ic       ��(	cC%�B=�Av*


acc��L>=���       ��2	��/�B=�Aw*

val_loss6�0?n�Z�       `/�#	�/�B=�Aw*

val_accWQ>��]O       ��-	��/�B=�Aw*

loss��0?�Q�d       ��(	��/�B=�Aw*


accCS>���       ��2	L4:�B=�Ax*

val_loss^�0?Q��U       `/�#	]5:�B=�Ax*

val_acc�k>R�hE       ��-	�5:�B=�Ax*

loss��0?���       ��(	;6:�B=�Ax*


acc�]O>zr�	       ��2	ޒA�B=�Ay*

val_loss6�0?��O�       `/�#	�A�B=�Ay*

val_acc�}b>l��       ��-	��A�B=�Ay*

loss��0?5��       ��(	��A�B=�Ay*


acc�Z>\&�4       ��2	O�I�B=�Az*

val_loss`1?/��       `/�#	?�I�B=�Az*

val_accd�]>��Y�       ��-	��I�B=�Az*

loss��0?���       ��(	3�I�B=�Az*


accQ�Y>�T       ��2	v�V�B=�A{*

val_lossa0?_�\       `/�#	�W�B=�A{*

val_acc�v_>�iN       ��-	:W�B=�A{*

loss��0?���       ��(	�W�B=�A{*


acc��]>z�W       ��2	e�c�B=�A|*

val_loss�`0?z�R�       `/�#	��c�B=�A|*

val_acc��L>���       ��-	l�c�B=�A|*

loss�+0?1=8       ��(	I�c�B=�A|*


acc�]>w�       ��2	�m�B=�A}*

val_loss�0?�s�p       `/�#	�m�B=�A}*

val_accбy>	^\�       ��-	hm�B=�A}*

loss�\0?iF|&       ��(	�m�B=�A}*


acc�2]>Զ��       ��2	��t�B=�A~*

val_loss@�1?-
t       `/�#	��t�B=�A~*

val_acc�z>��       ��-	/�t�B=�A~*

lossH0?}Y��       ��(	��t�B=�A~*


acc�}d>��?       ��2	]�|�B=�A*

val_lossS�0?���
       `/�#	8�|�B=�A*

val_acc�v_>���T       ��-	¤|�B=�A*

lossL0?.�?�       ��(	&�|�B=�A*


acc��c>�c)o       QKD	���B=�A�*

val_loss�0?����       ��2	����B=�A�*

val_acc��>(�ւ       �	����B=�A�*

loss�i0?�MP_       ��-	&���B=�A�*


acc��_>
j�        QKD	���B=�A�*

val_loss��0?V��x       ��2	���B=�A�*

val_acck�R>E�g�       �	���B=�A�*

loss�B0?�I�S       ��-	��B=�A�*


acc%\c>��K3       QKD	~��B=�A�*

val_lossH0?�gɗ       ��2	ǁ��B=�A�*

val_acc�g>�xQJ       �	����B=�A�*

loss��/?þ}       ��-	񄜘B=�A�*


acc�ch>�Gy�       QKD	�ݨ�B=�A�*

val_loss(�/?6��       ��2	Dߨ�B=�A�*

val_acc�$�>�+Ľ       �	�ߨ�B=�A�*

loss��/?//޼       ��-	7ਘB=�A�*


accGHk>N=W       QKD	�2��B=�A�*

val_loss��/?ƫ�2       ��2	4��B=�A�*

val_acc���>(� �       �	�4��B=�A�*

lossϠ/?�XO�       ��-	�4��B=�A�*


accӎo>��~z       QKD	����B=�A�*

val_lossz�/?���S       ��2	����B=�A�*

val_accx�a>��ls       �	���B=�A�*

loss��/?ٌ�       ��-	f���B=�A�*


acc]�p>	T	       QKD	5�ǘB=�A�*

val_loss��/?�<PG       ��2	��ǘB=�A�*

val_acc'�>�
�4       �	��ǘB=�A�*

loss�k/?e-�Z       ��-	��ǘB=�A�*


acc"q>"#��       QKD	1�ΘB=�A�*

val_lossV�/?��	       ��2	ȁΘB=�A�*

val_accyn�>�uO       �	k�ΘB=�A�*

loss�e/?ǌ_�       ��-	�ΘB=�A�*


acc�/p>>R       QKD	ɏטB=�A�*

val_lossȅ/?,CVL       ��2	ՑטB=�A�*

val_acc1�s>�8�       �	��טB=�A�*

loss�/?J�>       ��-	y�טB=�A�*


acc��r>�
�       QKD	r��B=�A�*

val_loss�/?̻7�       ��2	���B=�A�*

val_acc�c>`�p!       �	.��B=�A�*

lossDs/?��G�       ��-	���B=�A�*


accpNo>���       QKD	/��B=�A�*

val_loss,0?@�)       ��2	���B=�A�*

val_acc%�>a��       �	M��B=�A�*

loss4\/?/Xv�       ��-	���B=�A�*


acc�q><�ɜ       QKD	�f�B=�A�*

val_loss��/?�UĜ       ��2	Dh�B=�A�*

val_acc;��>h�-       �	�h�B=�A�*

loss�A/?�BF�       ��-	/i�B=�A�*


acc�r>O�T�       QKD	����B=�A�*

val_loss'?/?��eo       ��2	@���B=�A�*

val_acc��> u�q       �	���B=�A�*

lossZ/?_Fm       ��-	����B=�A�*


acc�Zw>�f@�       QKD	���B=�A�*

val_lossgz/?ZE��       ��2	���B=�A�*

val_acc3l�>�w�x       �	D��B=�A�*

loss�/?�[5&       ��-	���B=�A�*


acc$Vt>E��       QKD	��B=�A�*

val_loss��.?}��>       ��2	��B=�A�*

val_acc�i�>"��       �	}	�B=�A�*

loss
/?�7�[       ��-	�	�B=�A�*


acc�Zw>���       QKD	���B=�A�*

val_lossp�.?��)�       ��2	}��B=�A�*

val_accyn�>���J       �	���B=�A�*

lossL�.?ԁK       ��-	Ե�B=�A�*


acc�=y>�{��       QKD	�� �B=�A�*

val_loss�8/?I��       ��2	�� �B=�A�*

val_acc�9}>/�c�       �	� �B=�A�*

loss��.?*�@�       ��-	�� �B=�A�*


acc��y>�g"N       QKD	6:)�B=�A�*

val_loss;�/?��^       ��2	�;)�B=�A�*

val_acc� r>��{       �	}<)�B=�A�*

lossV/?�N�       ��-	=)�B=�A�*


acc�5t>s���       QKD	�J7�B=�A�*

val_loss�.?��]x       ��2	vS7�B=�A�*

val_acc��>��       �	�U7�B=�A�*

lossi�.?�R_       ��-	�W7�B=�A�*


acc͹v>h1�       QKD	;�A�B=�A�*

val_lossP�.?�7�       ��2	z�A�B=�A�*

val_acc'�>�v��       �	;�A�B=�A�*

loss��.?u��       ��-	�A�B=�A�*


accg>sc�,       QKD	:xJ�B=�A�*

val_loss�/?�       ��2	�yJ�B=�A�*

val_accw�t>��<       �	yzJ�B=�A�*

loss�.?&�       ��-	A{J�B=�A�*


acc��t>��/
       QKD	ikS�B=�A�*

val_loss~�.?����       ��2	nlS�B=�A�*

val_acc���>��k�       �	�lS�B=�A�*

loss�.?��       ��-	XmS�B=�A�*


accuy>�A       QKD	�uY�B=�A�*

val_loss �.?�>n	       ��2	�vY�B=�A�*

val_accc�>R�W�       �	`wY�B=�A�*

loss"�.?sdN       ��-	�wY�B=�A�*


acc��{>�ۀd       QKD	��^�B=�A�*

val_loss)�.?�=z       ��2	��^�B=�A�*

val_accr2�>���       �	��^�B=�A�*

loss��.?Va�A       ��-	_�^�B=�A�*


acc_�s>f]O�       QKD	�ph�B=�A�*

val_loss/?�S       ��2	 sh�B=�A�*

val_acc�}b>l&m�       �	�sh�B=�A�*

loss��.?��]       ��-	�th�B=�A�*


acc�x>>ː{       QKD	��p�B=�A�*

val_loss{�.?��M       ��2	
�p�B=�A�*

val_accr2�>�ԩ       �	��p�B=�A�*

loss�</?�	R       ��-	V�p�B=�A�*


acc��{>��.�       QKD	�iw�B=�A�*

val_lossP�.?N��V       ��2	�jw�B=�A�*

val_acc�+�>��Hk       �	Mkw�B=�A�*

loss��.?}��U       ��-	�kw�B=�A�*


acc�&>)�-+       QKD	)�~�B=�A�*

val_lossUF/?R=C|       ��2	��~�B=�A�*

val_acc�>e:�       �	��~�B=�A�*

lossʌ.?��%g       ��-	��~�B=�A�*


accV��>���       QKD	伈�B=�A�*

val_loss�?0?�6n�       ��2	Q���B=�A�*

val_acc])�>@S�       �	8���B=�A�*

loss��.?��v       ��-	�B=�A�*


acc�> ���       QKD	�0��B=�A�*

val_losst�.?j=�d       ��2	U2��B=�A�*

val_acc�-�>j�D�       �	3��B=�A�*

loss��.?�i       ��-	�3��B=�A�*


acc2�}>�y��       QKD	���B=�A�*

val_lossc.?�s-       ��2	����B=�A�*

val_acc%�>~P&|       �	��B=�A�*

lossP.?�ːq       ��-	�Ü�B=�A�*


accx�>3t\c       QKD	(~��B=�A�*

val_lossmD/?�FD8       ��2	����B=�A�*

val_acc�i�>#��       �	{���B=�A�*

loss<.?�Ev       ��-	����B=�A�*


acc��~>��       QKD	wذ�B=�A�*

val_lossn�.?�
�       ��2	'ڰ�B=�A�*

val_accٵ�>���       �	�ڰ�B=�A�*

loss��.?*l�h       ��-	4ݰ�B=�A�*


acc_�>���       QKD	�T��B=�A�*

val_lossS .?��w�       ��2	1Y��B=�A�*

val_accHu�>����       �	�Z��B=�A�*

loss�;.?�u�.       ��-	5\��B=�A�*


acc䔀>#��       QKD	ի��B=�A�*

val_loss�.?4۠�       ��2	T���B=�A�*

val_accHu�>i݆�       �	)���B=�A�*

loss<�-?_�       ��-	����B=�A�*


accQ�~>��׏       QKD	��̙B=�A�*

val_lossv�-?�^)       ��2	,�̙B=�A�*

val_acc�w�>����       �	��̙B=�A�*

loss�-?V���       ��-	A�̙B=�A�*


acc���>� �9       QKD	p�ԙB=�A�*

val_loss`.? �E�       ��2	x�ԙB=�A�*

val_acc�0y>w�=�       �	��ԙB=�A�*

loss��-?��T       ��-	l�ԙB=�A�*


acc_�>3)�       QKD	�ݙB=�A�*

val_loss�.?	WƜ       ��2	�ݙB=�A�*

val_acc���>�g�       �	�
ݙB=�A�*

loss��-?\�S�       ��-	�ݙB=�A�*


acc=v�>."e�       QKD	�H�B=�A�*

val_loss/?�f�       ��2	�L�B=�A�*

val_accj��>k\��       �	�N�B=�A�*

loss��.?����       ��-	�P�B=�A�*


accm}>@d��       QKD	�B=�A�*

val_loss�$.?-�$       ��2	?�B=�A�*

val_acc
��>t��G       �	��B=�A�*

lossl.?��۬       ��-	:�B=�A�*


acc)%~>
ԃ�       QKD	�g��B=�A�*

val_loss��-?*�FM       ��2	�h��B=�A�*

val_acc\~�>?���       �	�i��B=�A�*

loss��-?�"��       ��-	�i��B=�A�*


acc_�>M�š       QKD	E���B=�A�*

val_lossC�-?^��p       ��2	����B=�A�*

val_acc,0�>��M       �	����B=�A�*

lossF-?F��       ��-	����B=�A�*


acce7�>1~��       QKD	��B=�A�*

val_loss�.?*�4       ��2	��B=�A�*

val_acc3l�>��       �	���B=�A�*

loss��-?O(       ��-	���B=�A�*


acc�&>�.�       QKD	�R�B=�A�*

val_loss��-?K�c&       ��2	�S�B=�A�*

val_accV�>�jY       �	T�B=�A�*

loss�-?��x5       ��-	�T�B=�A�*


acc��~>�eQd       QKD	f1�B=�A�*

val_loss|�-?z]�       ��2	�3�B=�A�*

val_acc�g�>+J��       �	84�B=�A�*

loss�u-?C��       ��-	�4�B=�A�*


acc���>�j�       QKD	�"�B=�A�*

val_loss��-?6$E�       ��2	#�B=�A�*

val_accj��>���=       �	T#�B=�A�*

lossѲ-?ڢ&�       ��-	#�B=�A�*


acc��~>)�2       QKD	S!0�B=�A�*

val_loss�--?���l       ��2	p"0�B=�A�*

val_acc
��>�&       �	�"0�B=�A�*

loss1-?�N�+       ��-	[#0�B=�A�*


acc7$�>��l�       QKD	68�B=�A�*

val_loss"�-?���       ��2	X78�B=�A�*

val_acc�w�>�S�~       �	88�B=�A�*

loss�-?���       ��-	�88�B=�A�*


acc��>;��-       QKD	2?�B=�A�*

val_loss�Y-?��[       ��2	2?�B=�A�*

val_accHu�>���       �	�?�B=�A�*

loss	4-?4��       ��-	?�B=�A�*


acc�~>3"�       QKD	68G�B=�A�*

val_loss!�-?���       ��2	�9G�B=�A�*

val_accj�e>�k'�       �	�:G�B=�A�*

loss��,?�o8]       ��-	&;G�B=�A�*


accx�>�\�       QKD	�zO�B=�A�*

val_loss�Y-?=���       ��2	��O�B=�A�*

val_acc�4�>��o       �	��O�B=�A�*

loss�*-?�i�       ��-	͐O�B=�A�*


acc�>/z�       QKD	�!Z�B=�A�*

val_loss-�,?���       ��2	-#Z�B=�A�*

val_acc�=�>���C       �	�#Z�B=�A�*

loss�,?�&�       ��-	$Z�B=�A�*


acce7�>ŋ�U       QKD	O�`�B=�A�*

val_loss�L-?���        ��2	q�`�B=�A�*

val_accc��>���\       �	��`�B=�A�*

loss�,?�W�_       ��-	`�`�B=�A�*


acc_�>W�D�       QKD	�i�B=�A�*

val_loss�M-?���+       ��2	>i�B=�A�*

val_acck��>	���       �	pi�B=�A�*

loss~,-?��M�       ��-	�i�B=�A�*


acc��>�_�       QKD	=Hp�B=�A�*

val_loss�-?򌮱       ��2	ZIp�B=�A�*

val_acc�fX>�1�       �	�Ip�B=�A�*

loss�,?�)       ��-	EJp�B=�A�*


acc$f�>H�=       QKD	�Tw�B=�A�*

val_loss4�-?��1i       ��2	\Vw�B=�A�*

val_acc���>��h       �	7Ww�B=�A�*

loss�-?�-Q       ��-	�Ww�B=�A�*


acc��z>,z\       QKD	k�~�B=�A�*

val_lossv-?iƯ�       ��2	�~�B=�A�*

val_acc])�>�ɟ�       �	V�~�B=�A�*

loss�0-?�D�       ��-	��~�B=�A�*


acc�=y>��*       QKD	���B=�A�*

val_loss��,?��{X       ��2	j���B=�A�*

val_acc�+�>=et       �	����B=�A�*

loss,�,?F�x       ��-	]���B=�A�*


accZE~>���u       QKD	���B=�A�*

val_loss��,?�{�       ��2	%	��B=�A�*

val_acc;��>0�       �	��B=�A�*

loss4�,?�(
�       ��-	���B=�A�*


acc��>V�>0       QKD	����B=�A�*

val_loss�,?��5       ��2	����B=�A�*

val_acc��>0�ɗ       �	���B=�A�*

lossQ�,?a;       ��-	����B=�A�*


acc=v�>�HV�       QKD	�Z��B=�A�*

val_loss_i,?w�P�       ��2	�[��B=�A�*

val_acc�w�>�Ϭ�       �	B\��B=�A�*

loss�z,?� l       ��-	�\��B=�A�*


acc}G�>�-r       QKD	��B=�A�*

val_loss�r-?� ��       ��2	���B=�A�*

val_acc
��>�2�       �	]��B=�A�*

loss|�,?ep�=       ��-	���B=�A�*


acc.ŀ>c�       QKD	ܶ��B=�A�*

val_lossW�,??R�       ��2	̷��B=�A�*

val_acc���>��Z       �	E���B=�A�*

loss��,?y��       ��-	����B=�A�*


acc���>ǐ�G       QKD	Vc��B=�A�*

val_lossS�,?P<�       ��2	Rd��B=�A�*

val_accs�>٥b	       �	�d��B=�A�*

loss�,?'aٚ       ��-	Ne��B=�A�*


acc��>�鱔       QKD	D�ŚB=�A�*

val_lossO�,?It�       ��2	H�ŚB=�A�*

val_acc�6�>��xo       �	£ŚB=�A�*

lossP,?�&��       ��-	#�ŚB=�A�*


acc_�>6{ r       QKD	�DΚB=�A�*

val_loss)�,?�{\       ��2	�EΚB=�A�*

val_accB�>jGP3       �	$FΚB=�A�*

loss/,?����       ��-	�FΚB=�A�*


accx��>��Y       QKD	��ؚB=�A�*

val_loss\e,?<��z       ��2	��ؚB=�A�*

val_acc�w�>i�P�       �	J�ؚB=�A�*

loss�,?>�:       ��-	��ؚB=�A�*


acc���>EH�       QKD	���B=�A�*

val_loss�_,?�6L�       ��2	��B=�A�*

val_acc���>p$��       �	���B=�A�*

lossl,?9PA�       ��-	
��B=�A�*


acc�ك>���       QKD	$��B=�A�*

val_lossMk,?���K       ��2	I��B=�A�*

val_accc��>:���       �	k��B=�A�*

loss:�+?�K��       ��-	Y��B=�A�*


acc��>iP
�       QKD	���B=�A�*

val_loss��,?͖h�       ��2	g��B=�A�*

val_acc�-�>�(       �	���B=�A�*

loss*�,?�8��       ��-	V��B=�A�*


acc�e~>�{n�       QKD	�Q �B=�A�*

val_loss�X,?� C       ��2	�R �B=�A�*

val_acc�4�>]�P�       �	uS �B=�A�*

loss�',?�       ��-	�S �B=�A�*


accn��>�ѱ       QKD	�F�B=�A�*

val_lossZ�,?��o�       ��2	#H�B=�A�*

val_acc<~>��	�       �	�H�B=�A�*

loss[,?�9��       ��-	=I�B=�A�*


acc��>tW�       QKD	��B=�A�*

val_lossAI,?y���       ��2	G�B=�A�*

val_acc�p�> �(       �	�B=�A�*

lossF ,?�L       ��-	��B=�A�*


acc4�>����       QKD	G��B=�A�*

val_lossm
,?���       ��2	���B=�A�*

val_acc��>s���       �	��B=�A�*

loss��+?eyv       ��-	v��B=�A�*


acc0
�>{�       QKD	�y&�B=�A�*

val_lossɼ+?�M�_       ��2	x~&�B=�A�*

val_acc'�>#6��       �	��&�B=�A�*

loss5�+?���|       ��-	��&�B=�A�*


accf|�>���       QKD	�-�B=�A�*

val_loss�>,?C �v       ��2	��-�B=�A�*

val_accO��>ޤI       �	o�-�B=�A�*

lossH�+?b�D2       ��-	��-�B=�A�*


acc�e~>��0�       QKD	��4�B=�A�*

val_loss�-?)6�       ��2	Z�4�B=�A�*

val_accB�>����       �	ٵ4�B=�A�*

loss��+?Й?       ��-	9�4�B=�A�*


acc��>���       QKD	/�;�B=�A�*

val_loss@V,?�I��       ��2	#�;�B=�A�*

val_acc�k>{��       �	��;�B=�A�*

loss
�+?�!y       ��-	"�;�B=�A�*


acc�d�>Kݖ�       QKD	�>D�B=�A�*

val_loss.,?F�ne       ��2	>@D�B=�A�*

val_acc�;�>P	v       �	�@D�B=�A�*

lossyd,?*3C       ��-	%AD�B=�A�*


acc_�>_�Я       QKD	��K�B=�A�*

val_lossJ-?Q?jz       ��2	�$L�B=�A�*

val_accŬ�>�d��       �	�&L�B=�A�*

loss�>,?h�H�       ��-	c'L�B=�A�*


acc�Ɓ>�l�{       QKD	�U�B=�A�*

val_losso�,?;���       ��2	�U�B=�A�*

val_acc��>�Z        �	kU�B=�A�*

loss8L,?A�L       ��-	U�B=�A�*


acc���>#3-�       QKD	�^�B=�A�*

val_loss��,?�Y�       ��2	�^�B=�A�*

val_accŬ�>����       �	�^�B=�A�*

loss/4,?5)(       ��-	�^�B=�A�*


accP4�>6:�       QKD	��e�B=�A�*

val_loss�+?KsV       ��2	��e�B=�A�*

val_acc%�>fd��       �	;�e�B=�A�*

loss��+?��g       ��-	��e�B=�A�*


acc���>fʹ9       QKD	q�p�B=�A�*

val_losss�+?���U       ��2	7�p�B=�A�*

val_acc|�>�TZ       �	 �p�B=�A�*

loss[�+?`��       ��-	��p�B=�A�*


acc�J�>�tG       QKD	
Mw�B=�A�*

val_loss�,?�F^       ��2	aOw�B=�A�*

val_accc��>�5�       �	�Rw�B=�A�*

loss��+?��S       ��-	~Sw�B=�A�*


acc�w�>�M�K       QKD	V��B=�A�*

val_lossr8,?4�-       ��2	V��B=�A�*

val_acc\~�>	5       �	؜�B=�A�*

lossL�+?�$}�       ��-	A��B=�A�*


acc}G�>�->�       QKD	����B=�A�*

val_loss��,?���E       ��2	裆�B=�A�*

val_acc'�>V��8       �	����B=�A�*

lossϞ+?��o       ��-	���B=�A�*


acc�ց>�l?w       QKD	I}��B=�A�*

val_loss��+?T       ��2	�~��B=�A�*

val_acc%�>0���       �	���B=�A�*

lossq+?�c^\       ��-	R���B=�A�*


acc�̅>�r.       QKD	����B=�A�*

val_loss�o+?̜��       ��2	���B=�A�*

val_acc|�>�lOp       �	����B=�A�*

lossj+?�"!2       ��-	���B=�A�*


acc���>#c��       QKD	����B=�A�*

val_loss.\,?R�S�       ��2	� �B=�A�*

val_accV�>tۋ       �	nà�B=�A�*

lossdL+?�s�K       ��-	�à�B=�A�*


acc&��>���i       QKD	�^��B=�A�*

val_loss�+?�j�%       ��2	�`��B=�A�*

val_acc���>��       �	�a��B=�A�*

loss�T+?��H       ��-	gb��B=�A�*


acc�(�>����       QKD	ׄ��B=�A�*

val_lossH�+?�0       ��2	����B=�A�*

val_acc#,w>yF6�       �	w���B=�A�*

loss<k+?�& #       ��-	܆��B=�A�*


acca*�>m�       QKD	����B=�A�*

val_loss�+?�ꋅ       ��2	8���B=�A�*

val_acc�y�>\$�       �	����B=�A�*

loss�)+?�-�:       ��-	'���B=�A�*


acc�Z�>�q�       QKD	R���B=�A�*

val_loss�f+?���c       ��2	����B=�A�*

val_accc��>Y�       �	����B=�A�*

lossI�*?b\�6       ��-	V���B=�A�*


accL�>r00       QKD	��ǛB=�A�*

val_lossI�+?wTc       ��2	�ǛB=�A�*

val_accyn�>@D�       �	��ǛB=�A�*

loss�+?<��       ��-	�ǛB=�A�*


accV�>|I��       QKD	��ΛB=�A�*

val_loss��+?��?6       ��2	ґΛB=�A�*

val_acc�-�>�D��       �	l�ΛB=�A�*

loss��*?x�;�       ��-	ޒΛB=�A�*


acc��>/r��       QKD	�m՛B=�A�*

val_lossNO+?�ھ~       ��2	Wy՛B=�A�*

val_acc�w�>k       �	9|՛B=�A�*

lossb+?�F.       ��-	~՛B=�A�*


acc Y�>ީs�       QKD	I�ܛB=�A�*

val_loss2�+?�0�       ��2	M�ܛB=�A�*

val_acc])�>]��       �	ýܛB=�A�*

loss�$+?����       ��-	'�ܛB=�A�*


acc���>Qi�&       QKD	F��B=�A�*

val_lossr+?Ħ��       ��2	K��B=�A�*

val_accj��>H�e�       �	6��B=�A�*

loss^�*?&,��       ��-	���B=�A�*


acc���>c(^       QKD	�D��B=�A�*

val_loss��*?DG8       ��2	RG��B=�A�*

val_acc���>\���       �	�H��B=�A�*

lossl�*?�T_       ��-	�I��B=�A�*


acc���>}7�       QKD	�C��B=�A�*

val_loss�+?]S��       ��2	E��B=�A�*

val_acc%�>�Z=
       �	�E��B=�A�*

loss��*?�G       ��-	 F��B=�A�*


acca*�>�Zy#       QKD	�B=�A�*

val_loss��*?{�
       ��2		�B=�A�*

val_acc�i�>(!7       �	��B=�A�*

loss��*?F�י       ��-	 �B=�A�*


acc��>�-       QKD	'��B=�A�*

val_lossE�*?b��       ��2	���B=�A�*

val_accyn�>�C	�       �	f��B=�A�*

loss�*?nm�       ��-	���B=�A�*


acc?��>n� S       QKD	g�B=�A�*

val_loss�+?q s|       ��2	�m�B=�A�*

val_acc�+�>ȟ?�       �	�o�B=�A�*

loss��*?�[�       ��-	mr�B=�A�*


acc Y�>Ǟ�:       QKD	���B=�A�*

val_lossW�+?{Hj        ��2	���B=�A�*

val_acc�;�>d|7�       �	e��B=�A�*

lossW�*?�D�       ��-	���B=�A�*


acc?��>M�xi       QKD	M��B=�A�*

val_loss�++?`e%�       ��2	���B=�A�*

val_accHu�>����       �	���B=�A�*

loss]1+?��       ��-	���B=�A�*


acc�>ٖ��       QKD	�(�B=�A�*

val_loss@�+?��       ��2	�(�B=�A�*

val_accr2�>x��       �	�(�B=�A�*

loss�`*?i��       ��-	u(�B=�A�*


acc��>���       QKD	��3�B=�A�*

val_lossͱ*?J�       ��2	Ɍ3�B=�A�*

val_acc�;�>�lJ�       �	y�3�B=�A�*

loss�i*?Z���       ��-	�3�B=�A�*


accI�>\�7�       QKD	�]=�B=�A�*

val_lossu�*?���       ��2	�_=�B=�A�*

val_accc�>��a#       �	N`=�B=�A�*

loss�@*?�q�v       ��-	�`=�B=�A�*


acc�܅>����       QKD	��D�B=�A�*

val_loss�+?�P       ��2	L�D�B=�A�*

val_acc���>K^x       �	��D�B=�A�*

loss�-*?#LH�       ��-	6�D�B=�A�*


acc4�>��[       QKD	P�M�B=�A�*

val_lossI!+?���       ��2	��M�B=�A�*

val_acc3��>��       �	?�M�B=�A�*

loss�k*?z���       ��-	��M�B=�A�*


acc�>�=�u       QKD	F�V�B=�A�*

val_loss��*?���=       ��2	��V�B=�A�*

val_acc|�>�       �	:�V�B=�A�*

loss�N*?��a       ��-	��V�B=�A�*


acc���>ug�       QKD	�]�B=�A�*

val_lossϺ*?�=��       ��2	]�B=�A�*

val_accUB�>�p       �	�]�B=�A�*

loss&H*?�?�       ��-	�]�B=�A�*


accf|�>�F�!       QKD	��c�B=�A�*

val_loss��*?��Q�       ��2	Ĺc�B=�A�*

val_acc�>��l;       �	E�c�B=�A�*

lossck*?.ѯ       ��-	кc�B=�A�*


acc?��>��u�       QKD	N�j�B=�A�*

val_loss�+?�Ξ�       ��2	W�j�B=�A�*

val_accO��> 뤮       �	еj�B=�A�*

loss�*?'��       ��-	1�j�B=�A�*


acc��>M���       QKD	Ōr�B=�A�*

val_loss0�*?�Rߖ       ��2	ړr�B=�A�*

val_acc���>�d�       �	Ŕr�B=�A�*

loss��*?���y       ��-	>�r�B=�A�*


acc�5�>����       QKD	��}�B=�A�*

val_loss�0*?�-�	       ��2	!�}�B=�A�*

val_acc@�><�=~       �	��}�B=�A�*

loss� *?X"��       ��-	~�}�B=�A�*


acc��>���T       QKD	�=��B=�A�*

val_loss,�*?ǛZ       ��2	�>��B=�A�*

val_acc3l�>���y       �	W?��B=�A�*

loss}�)?��m�       ��-	�?��B=�A�*


acc�z�>�O       QKD	�͍�B=�A�*

val_loss��+?��       ��2	ύ�B=�A�*

val_acc	�k>G��       �	�ύ�B=�A�*

loss/�*?6%d�       ��-	�ύ�B=�A�*


acc$f�>Z���       QKD	����B=�A�*

val_loss�>*?��ݙ       ��2	����B=�A�*

val_acc3��>�⮫       �	���B=�A�*

loss�*?����       ��-	����B=�A�*


acc�Ɓ>f+�;       QKD	|���B=�A�*

val_loss�Y*?u��       ��2	Ϲ��B=�A�*

val_acc+l>cB	�       �	Z���B=�A�*

lossL�)?�6��       ��-	˺��B=�A�*


acc�Z�>��6�       QKD	����B=�A�*

val_lossDM*?J	�       ��2	����B=�A�*

val_acc�=�>i��       �	L���B=�A�*

loss�p*?N���       ��-	����B=�A�*


acc��>{�~�       QKD	m¬�B=�A�*

val_losssC*?o���       ��2	�ì�B=�A�*

val_accyn�>cS��       �	Ĭ�B=�A�*

loss��)?�m�       ��-	Ŭ�B=�A�*


acc<�>snŪ       QKD	ӿ��B=�A�*

val_loss�*?��O       ��2	���B=�A�*

val_acc�6�>��H       �	����B=�A�*

loss��)?U�=�       ��-	����B=�A�*


acc<�>N�D       QKD	�ƽ�B=�A�*

val_loss�I*?���       ��2	�ǽ�B=�A�*

val_acc��x>=��       �	XȽ�B=�A�*

loss{�)?��X	       ��-	�Ƚ�B=�A�*


acc�j�>qv�a       QKD	W?ƜB=�A�*

val_lossL*?���       ��2	�@ƜB=�A�*

val_accs�>?�gh       �	[AƜB=�A�*

loss}�)?V�?<       ��-	�AƜB=�A�*


accRy�>)R7%       QKD	�͜B=�A�*

val_loss8�*?΅��       ��2	��͜B=�A�*

val_acc�6�>��%       �	v�͜B=�A�*

lossH�)?m�L�       ��-	��͜B=�A�*


accRy�>U��       QKD	��לB=�A�*

val_loss��)?m�       ��2	��לB=�A�*

val_acc���>�
��       �	&�לB=�A�*

loss�x)?�}�       ��-	7�לB=�A�*


acc���>���^       QKD	��ޜB=�A�*

val_loss>P*?`�N�       ��2	��ޜB=�A�*

val_acc���>	u�        �	�ޜB=�A�*

loss��)?K0x&       ��-	v�ޜB=�A�*


accD�>�t+       QKD	w��B=�A�*

val_loss�+?ɬ�       ��2	��B=�A�*

val_acc>G�2       �	���B=�A�*

loss��)?�+a       ��-	���B=�A�*


acca*�>�N�       QKD	���B=�A�*

val_loss�,?f�       ��2	��B=�A�*

val_acc�>�H{�       �	���B=�A�*

lossB*?Ԣ��       ��-	��B=�A�*


acc�j�>��(�       QKD	����B=�A�*

val_loss�)?�^x       ��2	����B=�A�*

val_acc��q>�NV�       �	����B=�A�*

loss�.*?�=       ��-	���B=�A�*


accj��>�=,       QKD	W��B=�A�*

val_loss|�*?����       ��2	B��B=�A�*

val_accO��>K��       �	B��B=�A�*

loss`�)?���:       ��-	��B=�A�*


acc&��>����       QKD	t��B=�A�*

val_loss$.*?�kB_       ��2	��B=�A�*

val_accs�>�Nu�       �	B��B=�A�*

loss��)?��g       ��-	��B=�A�*


accpۄ>�f       QKD	\��B=�A�*

val_lossX)*?��g�       ��2	[��B=�A�*

val_acc��>��PO       �	̖�B=�A�*

loss�b)?�o�I       ��-	(��B=�A�*


acc�܅>�{@�       QKD	L��B=�A�*

val_loss�&*?۳��       ��2	L �B=�A�*

val_acc���>�       �	� �B=�A�*

loss��)?F��D       ��-	*�B=�A�*


acc Y�>XH5�       QKD	$��B=�A�*

val_lossR}*?}�       ��2	5��B=�A�*

val_acc])�>HE�       �	���B=�A�*

loss��)?��k�       ��-	
��B=�A�*


acc�ց>��       QKD	2Z'�B=�A�*

val_loss��*?��U       ��2	�['�B=�A�*

val_acc*�~>���       �	p\'�B=�A�*

loss-�)?�0D�       ��-	�\'�B=�A�*


accz:�>މA�       QKD	;�1�B=�A�*

val_losst�)?�uU�       ��2	֏1�B=�A�*

val_acc%�>���p       �	��1�B=�A�*

loss�)?PW��       ��-	)�1�B=�A�*


acc���>��\"       QKD	��6�B=�A�*

val_loss[*?�@�       ��2	!�6�B=�A�*

val_acc>�Efs       �	��6�B=�A�*

loss7,)?����       ��-	B�6�B=�A�*


acc,��>A�E       QKD	7�<�B=�A�*

val_loss��)?Q�s       ��2	��<�B=�A�*

val_acc])�>��I�       �	��<�B=�A�*

loss:�)?כ]       ��-	X�<�B=�A�*


acc��>z�V�       QKD	��B�B=�A�*

val_loss��)?+yf�       ��2	��B�B=�A�*

val_accٵ�>��       �	M�B�B=�A�*

lossKL)?ޒC2       ��-	��B�B=�A�*


accS��>�&#�       QKD	��J�B=�A�*

val_loss-�)?�а       ��2	ǵJ�B=�A�*

val_acc�;�>8�       �	_�J�B=�A�*

loss�)?�H�b       ��-	��J�B=�A�*


acc�}�>@2H       QKD		�R�B=�A�*

val_lossޑ)?d��       ��2	K�R�B=�A�*

val_acc%�>{gI�       �	��R�B=�A�*

lossD)?���       ��-	"�R�B=�A�*


acc���>��V�       QKD	$�Z�B=�A�*

val_lossA>*?�R�Y       ��2	F�Z�B=�A�*

val_accL>>��,       �	��Z�B=�A�*

lossb)??b׬       ��-	�Z�B=�A�*


accNl�>�       QKD	��c�B=�A�*

val_loss3�)?�be�       ��2	��c�B=�A�*

val_acc���>5�6       �	H�c�B=�A�*

loss�*)?��0�       ��-	��c�B=�A�*


acc�8�>�H       QKD	U�o�B=�A�*

val_loss�q)?�WI6       ��2	��o�B=�A�*

val_acc���>���|       �	��o�B=�A�*

loss0�(?�f,Z       ��-	D�o�B=�A�*


acc���>�       QKD	J&w�B=�A�*

val_loss�~*?�<�=       ��2	J'w�B=�A�*

val_accj��>M��       �	�'w�B=�A�*

loss�s)?����       ��-	(w�B=�A�*


acc���>؞�       QKD	x��B=�A�*

val_loss*?^Q       ��2	���B=�A�*

val_acc���>2       �	=��B=�A�*

lossā)?]�;l       ��-	���B=�A�*


acc=v�>?Ă        QKD	���B=�A�*

val_loss�w)?�qjy       ��2	T��B=�A�*

val_acc�;�>�p��       �	���B=�A�*

loss�)?x�D       ��-	B��B=�A�*


acc0
�>B:D\       QKD	dꎝB=�A�*

val_loss�})?����       ��2	�뎝B=�A�*

val_acc�y�>w���       �	쎝B=�A�*

lossl�(?�{��       ��-	�쎝B=�A�*


acc�܅>�m�       QKD	h@��B=�A�*

val_loss;�)?�NO�       ��2	_A��B=�A�*

val_acc�4�>�a       �	�A��B=�A�*

loss�)?����       ��-	)B��B=�A�*


acc�W�>q�~       QKD	���B=�A�*

val_loss!)?�       ��2	����B=�A�*

val_acc:��>��K       �	U���B=�A�*

loss �(?���       ��-	����B=�A�*


acc��>��]�       QKD	����B=�A�*

val_loss�4*?��:O       ��2	����B=�A�*

val_accV{a>�|�       �	����B=�A�*

lossr�(?���H       ��-	ꖧ�B=�A�*


acc���>���A       QKD	E���B=�A�*

val_loss��)?��A�       ��2	����B=�A�*

val_acc���>'���       �	����B=�A�*

loss]�)?����       ��-	ӽ��B=�A�*


acc��>%���       QKD	�&��B=�A�*

val_lossn�(?��1�       ��2	4,��B=�A�*

val_accV�>���h       �	M-��B=�A�*

loss�(?  �       ��-	�-��B=�A�*


acc���>�w#%       QKD	IÝB=�A�*

val_loss)?)Q��       ��2	JÝB=�A�*

val_acc���>�.�;       �	�JÝB=�A�*

loss�y(?�-��       ��-	�JÝB=�A�*


acc���>�       QKD	��̝B=�A�*

val_loss��)?(�X�       ��2	��̝B=�A�*

val_acc\~�>P��       �	M�̝B=�A�*

loss��(?&�Hb       ��-	��̝B=�A�*


acc���>8`�       QKD	w֝B=�A�*

val_loss�m)?:�o;       ��2	�֝B=�A�*

val_acc@�>A&'�       �	�֝B=�A�*

loss3�(?�Q�Y       ��-	I֝B=�A�*


acc���>4z�       QKD	dߝB=�A�*

val_loss�(? �       ��2	�fߝB=�A�*

val_acc|�>q�        �	�gߝB=�A�*

lossL�(?����       ��-	�hߝB=�A�*


acc�W�>����       QKD	�/�B=�A�*

val_loss��(?N�~�       ��2	I1�B=�A�*

val_acc|�>09n       �	�1�B=�A�*

loss�>(?�       ��-	H2�B=�A�*


acc5\�>����       QKD	V�B=�A�*

val_loss��(?qȥ       ��2	9�B=�A�*

val_acc�D�>{]�       �	��B=�A�*

loss�](?wD�)       ��-	Y�B=�A�*


acc�Ƀ>3*��       QKD	�l��B=�A�*

val_loss��(?����       ��2	�m��B=�A�*

val_acc|�>��8�       �	Pn��B=�A�*

loss=(?�:       ��-	�n��B=�A�*


accW˄>� �]       QKD	�U�B=�A�*

val_lossm�(?8�	�       ��2	KW�B=�A�*

val_acc>��>�       �	�W�B=�A�*

loss3(?�ԫF       ��-	hX�B=�A�*


acc�m�>yP�e       QKD	3��B=�A�*

val_loss��(?Ƹ�V       ��2	���B=�A�*

val_acc���>���-       �	���B=�A�*

lossJT(?
�>�       ��-	{��B=�A�*


acc�ކ>M�ĺ       QKD	M��B=�A�*

val_loss��(?a�       ��2	j��B=�A�*

val_acc�6�>x��       �	���B=�A�*

lossx6(?ى��       ��-	4��B=�A�*


acc��>'�J       QKD	1G�B=�A�*

val_loss$�(?<�J�       ��2	�H�B=�A�*

val_acc�6�>��I�       �	ZI�B=�A�*

loss�W(?iTz~       ��-	J�B=�A�*


acc���>n��g       QKD	��%�B=�A�*

val_lossZp(? ���       ��2	'�%�B=�A�*

val_acc|�>Ҩ�S       �	��%�B=�A�*

loss�(?p��8       ��-	P�%�B=�A�*


accRy�>�+ga       QKD	�,�B=�A�*

val_lossV�(?j�       ��2	�,�B=�A�*

val_acc�p�>�O"S       �	@,�B=�A�*

loss��'?+�4       ��-	�,�B=�A�*


acc��>��V�       QKD	et1�B=�A�*

val_lossƂ(?�˂m       ��2	�u1�B=�A�*

val_acc�;�>��)�       �	qv1�B=�A�*

lossyJ(? ;�-       ��-	�v1�B=�A�*


acc��>~�2       QKD	�7�B=�A�*

val_lossFB)?��9       ��2	�7�B=�A�*

val_accs�>��l_       �	h7�B=�A�*

lossy)(?Ѯ�{       ��-	�7�B=�A�*


acc9i�>�       QKD	u�>�B=�A�*

val_loss�(?l��       ��2	��>�B=�A�*

val_acc���>��84       �	��>�B=�A�*

loss��(?���       ��-	C�>�B=�A�*


acc.ŀ>Q�#