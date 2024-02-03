# Vision Transformer Implementation

Research Paper: [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)

# overview of the paper
- figure 1: viual overview of the architecture
- four equations: math equations which define the function ofeach layer/block
- table 1/3: diffrente hyperparamters for the architecture/trianing.
- text

### figure 1
<!-- ![Alt text](image-2.png) -->
![image-2.png](attachment:image-2.png)
* embedding - learnable representation (start with random numbers and improve them overtime) 
* mlp = multilevel perceptron


### four equations
<!-- ![Alt text](image-1.png) -->
![image-1.png](attachment:image-1.png)

Equaion1:
* embedding - learnable representation (start with random numbers and improve them overtime) 
* path embeddings - patches of original image - pink
* position embeddings - to retain positional information -  purple
* class token embedding - to perform classification (*/pink)

```python
x_input = [learnable_class_token, image_patch_embeddings_1, image_patch_embeddings_2,...image_patch_embeddings_N]+
            [learnable_class_token_pos, image_patch_1_embeddings_pos, image_patch_2_embeddings_pos,...image_patch_N_embeddings_pos]
```

Equation2:
* MSA = Multi-Head self attention (Multi-Head Attension)
* LN = LayerNorm (Norm))
* zv(l-1) = input before LN block, adding residual connections (+)

```python
x_output_MSA_block = MSA_layer(LN_layer(x_input)) + x_input 

```

Equation3:
* MLP = MultiLayer Perceptron
* LN = LayerNorm
* z'v(l) = input before LN block, adding residual connections (+)

```python
x_output_MLP_block = MLP_layer(LN_layer(x_output_MSA_block)) + x_output_MSA_block
```

Equation4:
* MLP = MultiLayer Perceptron - a nerual network with x no. of layers
* MLP = one hidden layer at training time
* MLP = single linear layer in fine-tuning time
* LN = LayerNorm

```python
y = Linear_layer(LN_layer(x_output_MLP_block)) 
```
or
```python
y = MLP(LN_layer(x_output_MLP_block)) 
```

### table 1/3:
<!-- ![Alt text](image-3.png) -->
![image-3.png](attachment:image-3.png)

#### all different sizes of the same model
#### ViT-B/16 - ViT-Base with image patch size 16x16

* layers = no. of transformers encoder layers
* hidden size $D$  - the embedding size throughout the architecture

                    - if we have embedding size of 768 means 

                    - each image patch that may be 16x16 

                    - is turned into a vector of size 768

                    - learnable vector*
                    
* MLP size - no. of hidden units/neurons in the MLP

            - if the MLP size is 3072 
            - then the no of hidden units in the MLP layer in 3072

* Heads - the number of heads within multihead self-attention layers

        - if heads = 12 we have 12 heads in MSA layer
        
<!-- > ![Alt text](image-4.png) -->
>![image-4.png](attachment:image-4.png)
        
        - denoted by h


>## Equation 1: split data into patches and creating the class, position and patch embedding 

### layers = input -> function -> output 

* what's the input shape?
* whats the output shape?

one of the biggest porblems in dl are misaligned tensor shapes

* Input shape: (224,224) -> single image -> (height, width, color channels)
* Output shape: 

* Input shape: $H*W*C$ [hieight,width, color channels]
* output shape: $N\times(P^2*C)$
- H = height
- W = width
- C = Color channels
- P = patchs ize
- N = number of patches = (height*width)/p^2
- D =  constant latent vector size = embedding dimension (see table 1)

```py
# exmaple values 
height = 224
width = 224
color_channels = 3
patch_size = 16

#calc the number of patches
num_patches = int((height *width) / (patch_size**2)) 
print(num_patches) 
```
Output:
196

input shape( single@D iamge): (224, 224, 3)
Output shape (single 1D sequence of pathces): (196, 768)


* Input shape: (224,224) -> single image -> (height, width, color channels)
* Output shape: (196,768) -> (number of patches, embedding_dimension)   ; embedding dimension = $D$ from table 1

## turning a single image into patches

Original Image:
<p align="center">
<img src="images/sushi_og.png" alt="drawing" style="width:600px;"/>
</p>

Number of parches per row: 14.0    
Number of patches per column: 14.0        
Patch size: 16 x 16 pixels

<p align="center">
<img src="images/sushi_patchified.png" alt="drawing" style="width:600px;"/>
</p>

14*14 = `196 patches`

## creating image patches and turning them into patch embeddings

* perhaps we could create the image patches and imge patch embedding ina  single step using 
`torch.nn.Conv2d()` and setting the kernel size and stride to `patch_size`.
- a convolutinal feature map is a laernable representation or `an embedding`

```py
#create conv2d layer to turn image into patches of learnable feature mas (embeddings))
from torch import nn
#set the patchpisze
patch_size = 16
#create a conv2d layers with hyperparameters from the ViT paper
conv2d = nn.Conv2d(in_channels=3,
                   out_channels=768, #D size from table 1 for ViT-Base, embedding size
                   kernel_size=patch_size,
                   stride=patch_size,
                   padding=0)
conv2d
```
Conv2d(3, 768, kernel_size=(16, 16), stride=(16, 16))

```py 
#paass the image thorugh the conv layer
image_out_of_conv = conv2d(image.unsqueeze(0)) #batchdimension  -> [batchsize, color channels, hieght, width]
print(image_out_of_conv.shape)
```
torch.Size([1, 768, 14, 14])

Now we've passed a single image into our conv2d layers we got 

```py
torch.Size([1, 768, 14, 14]) # [batchsize, embedding_dim, feature_map_height, feature_map_width]
```

```py 
image_out_of_conv.requires_grad
```
True

showing random convolutional feature maps from indexes: [683, 599, 74, 343, 635]

<p align="center">
<img src="images/conv_index.png" alt="drawing" style="width:600px;"/>
</p>

## flattening the patch embedding/ feature map with torch.nn.flatten
right now we've got a series of conv faeture maps (patch embeddings) that we want to flatten into a sequence of patch embeddings to satusfy the criteria of the
ViT 

torch.Size([1, 768, 14, 14]) -> [batch_size, embeddingdim, feature_map_height, feature_map_width]

### want [batch_size, num_of_patches, embeddingdim]

```py 
flatten_layer = nn.Flatten(start_dim=2, 
                           end_dim=3)
flatten_layer(image_out_of_conv).shape #the order is still not w
```
torch.Size([1, 768, 196])

Original image shape: torch.Size([3, 224, 224])
image feature map (patches) shape: torch.Size([1, 768, 14, 14])
Flattened image feature map shape : torch.Size([1, 768, 196])

torch.Size([1, 196, 768]) -> [batchsize, num of patches, embedding dimension]

## ^embedding vector that one of our images is represented by, 768 of these flattened vectors
## these are learnable ie they are update upon training


<p align="center">
<img src="images/single_feature_flattened.png" alt="drawing" style="width:600px;"/>
</p>

## Turning the ViT patch embedding layer into a pytorch modeule
we want this module to do a few things.
1. Create a class called PatchEmbedding that inherits from nn.Module
2. Initialize with appropriate hyperparameters, such a schannels, embedding dimension, patch_size.
3. create a layer to turn an imamge into embedded patches using nn.Conv@d().
4. create a layer to flatten the feautre maps of the output of the layer in 3.
5. define forward() that defines the forward computations (eg. pass through layer from 3 to 4)
6. make sure the output shape of the layer reflects the required output shape of the patch embedding.

```py
#1. create a class called Patchembedding
class PatchEmbedding(nn.Module):
    #2. initialize the layer owth appropriate hyperparamters
    def __init__(self,
                 in_channels: int=3,
                 patch_size: int=16,
                 embedding_dim: int=768): #from table 1 for ViT_base
        super().__init__()
        self.patch_size = patch_size
        #3. creata a layer to turn an image into embedded patches
        self.patcher = nn.Conv2d(in_channels=in_channels,
                                 out_channels=embedding_dim,
                                 kernel_size=patch_size,
                                 stride=patch_size,
                                 padding=0)
        #4. create a layer to flatten feature maps outputs of conv2d
        self.flatten = nn.Flatten(start_dim=2,
                                  end_dim=3)
        
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        #creata assertion to mkae sure the image resolution is compatable with the patch size
        image_resolution = x.shape[-1]
        assert image_resolution % patch_size == 0, f"Input image size must be divisible by patchsize, image shape: {image_resolution}, patch_size = {self.patch_size}"
        
        #perfrm forawrd pass
        x_patched = self.patcher(x)
        x_flattened = self.flatten(x_patched)
        #6. make sure thereturned sequence embeddign dim are in the righ torder [batchsize,no of patches, embdedding dim]
        return x_flattened.permute(0,2,1)


patchify = PatchEmbedding(in_channels=3,
                          patch_size=16,
                          embedding_dim=768)
# pass a single image thorugh the patch embedding layer
print(f"input image size: {image.unsqueeze(0).shape}")
patch_embedded_image = patchify(image.unsqueeze(0)) # add an extra batchdim
print(f"output patch embedding sequence shape: {patch_embedded_image.shape}")
```

input image size: torch.Size([1, 3, 224, 224])
output patch embedding sequence shape: torch.Size([1, 196, 768])


## creating the class token embedding 
* want to prepend a learnable class token to the start of the patch embedding
* in order to perform classificaiton we use the standard approach of adding an extra learnable classification token 


```py
patch_embedded_image.shape
```
torch.Size([1, 196, 768])

after we prepend the class token it should become [1,197, 768]

```py
#get the batchsize and embedding dimension 
batch_size = patch_embedded_image.shape[0]
embedding_dimension = patch_embedded_image.shape[-1]
batch_size, embedding_dimension

#create class token embedding as a laernable paramter that shares the same size as the embedding dimension (D)
class_token = nn.Parameter(torch.ones(batch_size, 1, embedding_dimension),
                           requires_grad=True)  # to make the paramter learnable its on by default
class_token.shape
```
torch.Size([1, 1, 768])

### USING ONES TO MAKE IT MORE VISIBLE for learning
### NOTe: use `randn` in practical use

```py
patch_embedded_image.shape
```

torch.Size([1, 196, 768])

```py
#add the class token embedding to the from of the patch embedding # From the paper
patch_embedded_image_with_class_embedding= torch.cat((class_token, patch_embedded_image),
                                                     dim=1) #no of patches dim 
print(f"{patch_embedded_image_with_class_embedding.shape} -> batch_size, class_token + no_of_patches, embedding_dim")
```
torch.Size([1, 197, 768]) -> batch_size, class_token + no_of_patches, embedding_dim

```py
patch_embedded_image_with_class_embedding
```
tensor([[[ 1.0000,  1.0000,  1.0000,  ...,  1.0000,  1.0000,  1.0000],
         [-0.2044,  0.1213, -0.3727,  ..., -0.0422, -0.1065, -0.1086],
         [-0.1035, -0.0072, -0.4738,  ..., -0.0445, -0.1931,  0.1067],
         ...,
         [-0.2448,  0.0557, -0.5102,  ..., -0.0727, -0.2336, -0.0618],
         [-0.1278,  0.0315, -0.3228,  ..., -0.0731, -0.1883, -0.0347],
         [-0.0428,  0.0037, -0.1469,  ..., -0.0665, -0.0937, -0.0178]]],
       grad_fn=<CatBackward0>)

## creating the position embedding

want to: create a series of 1d learnable positionembedding and to add them to the sequence of patch embeddings

```py
#view the sequence of patch embeddinigs withthe prepended class embeddings
patch_embedded_image_with_class_embedding, patch_embedded_image_with_class_embedding.shape
```
(tensor([[[ 1.0000,  1.0000,  1.0000,  ...,  1.0000,  1.0000,  1.0000],
          [-0.2044,  0.1213, -0.3727,  ..., -0.0422, -0.1065, -0.1086],
          [-0.1035, -0.0072, -0.4738,  ..., -0.0445, -0.1931,  0.1067],
          ...,
          [-0.2448,  0.0557, -0.5102,  ..., -0.0727, -0.2336, -0.0618],
          [-0.1278,  0.0315, -0.3228,  ..., -0.0731, -0.1883, -0.0347],
          [-0.0428,  0.0037, -0.1469,  ..., -0.0665, -0.0937, -0.0178]]],
        grad_fn=<CatBackward0>),
 torch.Size([1, 197, 768]))

```py
 #calculate the N no of  patches
num_patches = int((height * width) / patch_size**2)
#get the embedding dimension
embedding_dimension = patch_embedded_image_with_class_embedding.shape[-1]
embedding_dimension

#create the learnable 1D position embedding
position_embedding = nn.Parameter(torch.ones(batch_size, #batchsize or 1
                                             num_patches+1, #from paper # and as we added the class token  we get 197 instead of 196
                                             embedding_dimension),
                                  requires_grad=True) #learnable so it gets updated during training
position_embedding, position_embedding.shape
```

(Parameter containing:
 tensor([[[1., 1., 1.,  ..., 1., 1., 1.],
          [1., 1., 1.,  ..., 1., 1., 1.],
          [1., 1., 1.,  ..., 1., 1., 1.],
          ...,
          [1., 1., 1.,  ..., 1., 1., 1.],
          [1., 1., 1.,  ..., 1., 1., 1.],
          [1., 1., 1.,  ..., 1., 1., 1.]]], requires_grad=True),
 torch.Size([1, 197, 768]))

 Add the position embedding to the patch and class token embeddding

 ```py
 patch_embedded_image_with_class_embedding, patch_embedded_image_with_class_embedding.shape
 ```
 (tensor([[[ 1.0000,  1.0000,  1.0000,  ...,  1.0000,  1.0000,  1.0000],
          [-0.2044,  0.1213, -0.3727,  ..., -0.0422, -0.1065, -0.1086],
          [-0.1035, -0.0072, -0.4738,  ..., -0.0445, -0.1931,  0.1067],
          ...,
          [-0.2448,  0.0557, -0.5102,  ..., -0.0727, -0.2336, -0.0618],
          [-0.1278,  0.0315, -0.3228,  ..., -0.0731, -0.1883, -0.0347],
          [-0.0428,  0.0037, -0.1469,  ..., -0.0665, -0.0937, -0.0178]]],
        grad_fn=<CatBackward0>),
 torch.Size([1, 197, 768]))

```py
patch_and_position_embedding = patch_embedded_image_with_class_embedding + position_embedding
patch_and_position_embedding, patch_and_position_embedding.shape
 ```

 (tensor([[[2.0000, 2.0000, 2.0000,  ..., 2.0000, 2.0000, 2.0000],
          [0.7956, 1.1213, 0.6273,  ..., 0.9578, 0.8935, 0.8914],
          [0.8965, 0.9928, 0.5262,  ..., 0.9555, 0.8069, 1.1067],
          ...,
          [0.7552, 1.0557, 0.4898,  ..., 0.9273, 0.7664, 0.9382],
          [0.8722, 1.0315, 0.6772,  ..., 0.9269, 0.8117, 0.9653],
          [0.9572, 1.0037, 0.8531,  ..., 0.9335, 0.9063, 0.9822]]],
        grad_fn=<AddBackward0>),
 torch.Size([1, 197, 768]))

 ## as you can see all the values increased by one as we created positional embeddding with `torch.ones`

## successfully added it to the patch_embeddings with class_token

## Equation 1 complete

>## equation 1
![Alt text](images/image-1.png)

```py
import helper_functions
#set the seeds
helper_functions.set_seeds()

#!set the patch ize: there are multiple patchsizes in the paper we use 16
patch_size = 16

#2. print the sahapes of the original image and get the image dimensions
print(f"Image tensor shape: {image.shape}")
height, width = image.shape[1], image.shape[2]

#3. get the image tensor and add a batch dimension
x = image.unsqueeze(0)
print(f"Input image shape: {x.shape}")

#4.create a patch embedding layer
patch_embedding_layer = PatchEmbedding(in_channels=3,
                                       patch_size=patch_size,
                                       embedding_dim=768) #inline with ViT-base in table 1
#5. pass input iamge through patchembedding layer
patch_embedding = patch_embedding_layer(x)
print(f"Patch embedding shape: {patch_embedding.shape}")
#6.creating class token embedding
batch_size = patch_embedding.shape[0]
embedding_dimension = patch_embedding.shape[-1]
class_token = nn.Parameter(torch.ones(batch_size,1,embedding_dimension), #using ones for laerning use randn for practical architecture
                          requires_grad=True)
print(f"class token embedding shape: {class_token.shape}")
#7. prepend the class token embedding to the patch embedding
patch_embedding_class_token = torch.cat((class_token, patch_embedding), dim=1)
print(f"patch_embedding with class token shape :{patch_embedding_class_token.shape}")
#8.create position embedding 
number_of_patches = int((height*width) / patch_size**2)
position_embedding = nn.Parameter(torch.ones(1, number_of_patches+1, embedding_dimension),
                                  requires_grad=True) #look at paper for plus 1
#9. add the posotion embedding to patch embedding with class otken
patch_and_position_embedding = patch_embedding_class_token + position_embedding
print(f"patch_and_position_embedding shape: {patch_and_position_embedding.shape}")
```
Image tensor shape: torch.Size([3, 224, 224])
Input image shape: torch.Size([1, 3, 224, 224])
Patch embedding shape: torch.Size([1, 196, 768])
class token embedding shape: torch.Size([1, 1, 768])
patch_embedding with class token shape :torch.Size([1, 197, 768])
patch_and_position_embedding shape: torch.Size([1, 197, 768])

># Equation 2.
![image.png](attachment:image.png)

## we finished embedding the patches 
## now we enter the transformer encoder
- consists of 2 main blocks of eq.2 and eq.3

![image-1.png](attachment:image-1.png)
### equation 2 has the MSA block
- for many of these layers like `MSA` pytorch has prebuilt function for the layer

#### from [Attention Is All You Need paper](https://arxiv.org/abs/1706.03762)
![swappy-20231003-183132.png](attachment:swappy-20231003-183132.png)

![swappy-20231003-183647.png](attachment:swappy-20231003-183647.png)

![swappy-20231003-185912.png](attachment:swappy-20231003-185912.png)

Q- Query, K-key, V-value are instances of the `same` sequence of vectors,

tells us which patch needs to pay how much `attention` to which other word or image patch in the same vector,

to find how much related 1 patch is to another

![swappy-20231003-190543.png](attachment:swappy-20231003-190543.png)

># Equation 2.
![image.png](attachment:image.png)

## we finished embedding the patches 
## now we enter the transformer encoder
- consists of 2 main blocks of eq.2 and eq.3

![image-1.png](attachment:image-1.png)
### equation 2 has the MSA block
- for many of these layers like `MSA` pytorch has prebuilt function for the layer

#### from [Attention Is All You Need paper](https://arxiv.org/abs/1706.03762)
![swappy-20231003-183132.png](attachment:swappy-20231003-183132.png)

![swappy-20231003-183647.png](attachment:swappy-20231003-183647.png)

![swappy-20231003-185912.png](attachment:swappy-20231003-185912.png)

Q- Query, K-key, V-value are instances of the `same` sequence of vectors,

tells us which patch needs to pay how much `attention` to which other word or image patch in the same vector,

to find how much related 1 patch is to another

![swappy-20231003-190543.png](attachment:swappy-20231003-190543.png)

## equation 2 (layer normalization) (LN block)
* ### layer norm

            - normalization technique to normalize the distributions of the intermediate layers,
             
            - it enables smoother gradients, faster training, and better generalization accuracy 
- Normalization - make everything have name mean and same std deviation
- mean and std dev are calculated over the last D dimension, where D is the dim of normalized shape

$D$ in our case is the embedding dimensions here [768]

![swappy-20231003-192146.png](attachment:swappy-20231003-192146.png)
       
        * when we normalize along the embedding dimension, it's like making all of the steps in the staircase to the same size

in docs:
- no of patches = sequence 
- features = embedding dimension

```py
## creating a multihead self attentionlayer in pytorch
class MultiHeadSelfAttentionBlock(nn.Module):
    """creates a multihead self attention block (MSA block)
    """
    def __init__(self,
                 embedding_dimension: int=768, #Hidden size D (embedding dimension) from table 1 for ViT _base
                 num_heads: int=12, #heads from table 1 for ViTbase
                 attn_dropout: int=0):#dropout is only used for dense layer/ fully coneccted/ linaer/ feed forward. but not for QKV projections 
                                        #ie we are not gonna use dropout for our MSA block
        super().__init__()
    #create the norm layer (LN)
        self.layer_norm = nn.LayerNorm( 
            normalized_shape=embedding_dimension#check pytroch docs
        )
        
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=embedding_dimension, # hidden size D (embedding dimension) from table 1 for ViT-base
            num_heads=num_heads, #no of MSA heads
            dropout=attn_dropout, #zero
            batch_first=True #batch_first – If True, then the input and output tensors are provided as (batch, no of patches or sequence, features or embedding dimension). Default: False (seq, batch, feature) #from pytorch docs
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor: # Q,K and V are x: they are different instances of the same vector
        x = self.layer_norm(x)
        attn_output, _ =  self.multihead_attn(query=x,
                                                key=x,
                                                value=x,
                                                need_weights=False)                  # u can also get attn_weights ie weights of the attention layer #check the docs.
        return attn_output

```
```py
# create an instance MSA block
multihead_self_attention_block = MultiHeadSelfAttentionBlock(
    embedding_dimension=768,
    num_heads=12,
    attn_dropout=0
)

#pass the patch and position image embedding sequence thourgh our MSA Block
patched_image_through_msa_block = multihead_self_attention_block(patch_and_position_embedding)
print(f"Input shape of MSA bock: {patch_and_position_embedding.shape}")
print(f"Output shape of MSA block; {patched_image_through_msa_block.shape}")
```
Input shape of MSA bock: torch.Size([1, 197, 768])
Output shape of MSA block; torch.Size([1, 197, 768])

## even tho the shapes haven't changed maybe the values changed??

```py
patch_and_position_embedding 
```
tensor([[[2.0000, 2.0000, 2.0000,  ..., 2.0000, 2.0000, 2.0000],
         [0.4842, 1.1071, 0.9806,  ..., 1.2911, 0.8060, 1.1216],
         [0.4619, 0.9047, 0.8579,  ..., 1.3663, 0.9612, 1.4155],
         ...,
         [0.4886, 1.0627, 0.8966,  ..., 1.3370, 0.7364, 1.2223],
         [0.6908, 1.0200, 0.9475,  ..., 1.2131, 0.8064, 1.1861],
         [0.8316, 1.0010, 0.9662,  ..., 1.0398, 0.9115, 1.1034]]],
       grad_fn=<AddBackward0>)

```py
#output
patched_image_through_msa_block
```
tensor([[[-0.1992, -0.1785,  0.0754,  ..., -0.3689,  0.8296, -0.4426],
         [-0.1836, -0.1788,  0.0768,  ..., -0.3568,  0.8418, -0.4648],
         [-0.1715, -0.1662,  0.0784,  ..., -0.3658,  0.8433, -0.4552],
         ...,
         [-0.1745, -0.1771,  0.0726,  ..., -0.3638,  0.8436, -0.4722],
         [-0.1707, -0.1792,  0.0704,  ..., -0.3669,  0.8461, -0.4732],
         [-0.1732, -0.1884,  0.0665,  ..., -0.3622,  0.8442, -0.4701]]],
       grad_fn=<TransposeBackward0>)

```py
patched_image_through_msa_block == patch_and_position_embedding
```
tensor([[[False, False, False,  ..., False, False, False],
         [False, False, False,  ..., False, False, False],
         [False, False, False,  ..., False, False, False],
         ...,
         [False, False, False,  ..., False, False, False],
         [False, False, False,  ..., False, False, False],
         [False, False, False,  ..., False, False, False]]])

## how do we add the skip/residual connection in this?

>## Equation 3: Multilayer perceptron block (MLP block)
![image-1.png](attachment:image-1.png)

### **MLP** contains two layers with a GELU non-linearlity(mentioned above)

    * MLP = a quite broad term for a block with a series of layer(s), ayers can be multiple or even only one hidden layer.
    * layers can mean: fully connected/ dense/ linear/ feed-forward, all are often similar names for the same thing. 
    In pytorch theyre called `torch.nn.Linear`.
    In tensorflow theyre called `tf.keras.dense
**MLP_size / number of hidden units of MLP** = 3072 for ViT-Base (from table 1)

![image-3.png](attachment:image-3.png)

### [what is **GELU** non linearity](https://paperswithcode.com/method/gelu) 
![swappy-20231003-225340.png](attachment:swappy-20231003-225340.png)

    * the standard Gaussian cumulative distribution function. The GELU nonlinearity weights inputs by their percentile, rather than gates inputs by their sign as in ReLUs (). 
    * Consequently the GELU can be thought of as a smoother ReLU.
    * applies the Gaussin Error Linear Units Functions (GELU)

**Dropout** is applied after every linear layer in MLP

    * value for dropout is available at table 3 Dropout = 0.1 for Vit/b-16

![swappy-20231003-231246.png](attachment:swappy-20231003-231246.png)

In pseudocode:
```py
#MLP block
x = layer norm -> linear layer -> non-linear layer -> dropout -> linear layer -> dropout
```

```py
class MLPBlock(nn.Module):
    def __init__(self,
                 embedding_dim: int=768,
                 mlp_size: int=3072,
                 dropout: float=0.1
                 ):
        super().__init__()
        #creat the norm layer (LN)
        self.layer_norm = nn.LayerNorm(
            normalized_shape=embedding_dim
        )
        
        #creat eh MLP
        self.mlp = nn.Sequential(
            nn.Linear(in_features=embedding_dim,
                      out_features=mlp_size), #project to a larger size # to hopfully capture some fore information 
            nn.GELU(),
            nn.Dropout(
                p=dropout
            ),
            nn.Linear(in_features=mlp_size,
                      out_features=embedding_dim), #squeeze it back to the embedding dimension 
            nn.Dropout(
                p=dropout
            )
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer_norm(x)
        x = self.mlp(x)
        return x
        #return self.mlp(self.layer_norm(x) #same as above# this will benefit from operator fusion 
```

### side track:  https://horace.io/brrr_intro.html || checkout torch.compile feature that leverages operator fusion mentioned above

```py
#create an instance of MLP block
mlp_block = MLPBlock(embedding_dim=768, #talble 1 value
                     mlp_size=3072, #table 1 value
                     dropout=0.1) #table 3 value

#pass ouot through the MLP block
patched_image_through_mlp_block = mlp_block(patched_image_through_msa_block)
print(f"Input shape of the MLP block: {patched_image_through_msa_block.shape}")
print(f"Output shape of the  MLP blcok: {patched_image_through_mlp_block.shape}")
```
Input shape of the MLP block: torch.Size([1, 197, 768])
Output shape of the  MLP blcok: torch.Size([1, 197, 768])

```py
#the variables have changed even though the values are the same
patched_image_through_mlp_block == patched_image_through_msa_block
```
tensor([[[False, False, False,  ..., False, False, False],
         [False, False, False,  ..., False, False, False],
         [False, False, False,  ..., False, False, False],
         ...,
         [False, False, False,  ..., False, False, False],
         [False, False, False,  ..., False, False, False],
         [False, False, False,  ..., False, False, False]]])

## this is not really the correct order of doing things as we havent added the residual connections yet
## we are just creating modules of the final product

># creating the transformer encoder (with residual connections)
The transfformer encoder is a combination of alternating blocks of MSA blocks and MLP blocks
LN is applied between every block, and residual connections after every block

![image.png](attachment:image.png)

**encoder** = turn a sequence into a laernable representation

**decoder** = turna learnable representation to some sort of sequence

encoders in NLP converts text(sequence) into numbers 
decoders decontructs encoded numbers into text (sequence) (in a translator)

### also known as Seq2Seq
seq2seq is a ml model usually used in NLP. This is also where our transformer came from ie NLP which has `encoders as well as decoders`

thats were our name transformer `encoder` comes from, 
since our goal is image `classification` instead of a decoder we use an `MLP` instead of an `Decoder`

* residual connnections = add alyer(s) input to its subsequent output, 
this enables the creation of deeper networks (prevents weights from getting too small ie preventing `spare networks` or `vanishing gradient problem`)

In pseudo:
```python
#transfomer encoder:
x_input (output/sequence of embedding patches) -> MSA_Block -> [MSA_Block + x_input (residual)] -> MLP_Block -> [MLP_block_output + (MSA_block_output + x_input) ] -> ...
```

>### method 1:coding in the transformer block (puting it all together)

```py
class TransformerEncoderBlock(nn.Module):
    """creates a transformer block instance"""
    def __init__(self,
                 embedding_dim:int=768, # hidden size D from Table 1 for ViT-Base
                 num_heads:int=12, # heads from Table 1 for ViT-Base
                 mlp_size:int=3072, # MLP size from Table 1 for ViT-Base
                 mlp_dropout:float=0.1, # dropout for dense layers from Table 3 for ViT-Base
                 attn_dropout:float=0): # dropout for attention layers
        super().__init__()

        # 3. Create MSA block (equation 2)
        self.msa_block = MultiHeadSelfAttentionBlock(embedding_dimension=embedding_dim,
                                                     num_heads=num_heads,
                                                     attn_dropout=attn_dropout)
        
        # 4. Create MLP block (equation 3)
        self.mlp_block =  MLPBlock(embedding_dim=embedding_dim,
                                   mlp_size=mlp_size,
                                   dropout=mlp_dropout)
        
    # 5. Create a forward() method  
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        
        # 6. Create residual connection for MSA block (add the input to the output)
        x =  self.msa_block(x) + x 
        # 7. Create residual connection for MLP block (add the input to the output)
        x = self.mlp_block(x) + x 
        
        return x
```

```py
#create an instance of TransformerEncoderBlock()
transformer_encoder_block = TransformerEncoderBlock()
transformer_encoder_block
```

TransformerEncoderBlock(
  (msa_block): MultiHeadSelfAttentionBlock(
    (layer_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
    (multihead_attn): MultiheadAttention(
      (out_proj): NonDynamicallyQuantizableLinear(in_features=768, out_features=768, bias=True)
    )
  )
  (mlp_block): MLPBlock(
    (layer_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
    (mlp): Sequential(
      (0): Linear(in_features=768, out_features=3072, bias=True)
      (1): GELU(approximate='none')
      (2): Dropout(p=0.1, inplace=False)
      (3): Linear(in_features=3072, out_features=768, bias=True)
      (4): Dropout(p=0.1, inplace=False)
    )
  )
)