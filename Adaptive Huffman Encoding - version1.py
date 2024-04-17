class Node:
  def __init__(self, tag , parent,lchild,rchild,weight):
    self.tag = tag
    self.parent = parent
    self.lchild = lchild
    self.rchild = rchild
    self.weight = weight
def BinConvert(node):
    code=""
    while(node.parent!=None):
      parent=node.parent
      if(parent.lchild==node):
        code+="0"
      else:
        code+="1"
      node=node.parent
    return code[::-1]

def Ascii(number):
    binary=""
    while(number!=0):
      binary+=str(number%2)
      number//=2
    length=len(binary)
    return('0'*(7-length)+binary[::-1])

def Convert_to_symbol(code):
  value=0
  for i in code:
    value=value*2+int(i)
  return chr(value)

def sibling_property_test(node):               #Checks sibling property and swaps left and right children if not satisfied
    if(node.parent.lchild.weight>node.parent.rchild.weight):
      node.parent.lchild,node.parent.rchild=node.parent.rchild, node.parent.lchild
      #print('swap happened')

def findflow(node):
  parent=node.parent
  while(parent!=None):
    if node.weight>parent.weight:
      node_parent,node_left,node_right=node.parent,node.lchild, node.rchild
      parent_parent,parent_left,parent_right=parent.parent,parent.lchild,parent.rchild
      node.parent,node.lchild,node.rchild=parent_parent,parent_left,parent_right
      parent.parent,parent.lchild,parent.rchild=node_parent,node_left,node_right
      sibling_property_test(parent)
    parent=parent.parent

def UpdateTree(symbol):  
  #print('tree getting updated')                     #Traverses through the tree to check if sibling property is satisfied and updates weight of parent chain
  while(symbol.parent!=None):
    symbol.parent.weight+=1
    findflow(symbol)
    sibling_property_test(symbol)
    symbol=symbol.parent

def AdaHuffEncode(input_array):
  global stored_dict
  global count
  count=0                                        #count keeps track of numbers to use for nodes of unknown labels
  output=""
  NYT=Node('NYT',None,None,None,0)               #Always the NYT Node corresponds to NODE ORDER 0
  for i in input_array:
    updated_flag=0                               #This flag helps us to identify if a symbol was updated or not
    if i not in stored_dict:                     #Condition True when the symbol is appearing for the first time in the tree
       updated_flag=1
       existing_nyt_code=BinConvert(NYT)
       count+=1
       setup_node=Node(i,None,None,None,1)
       stored_dict[i]= setup_node

       if NYT.parent==None:                                            #This means that this is root node
          ROOT_NODE=Node('N'+str(count),None,NYT,setup_node,1)           #Setup root node
          NYT.parent=stored_dict[i].parent=ROOT_NODE                   #Map parents to the new node and NYT node

       else:
          parent=NYT.parent
          replace_node=Node('N'+ chr(count),parent,NYT,stored_dict[i],0) #Spanning the NYT node into two nodes: left-NYT and right- Symbol
          parent.lchild=NYT.parent=stored_dict[i].parent=replace_node    #Updating the flow of the tree



    if(updated_flag==1):
        output+=existing_nyt_code+Ascii(ord(i))
    else:
        output+=BinConvert(stored_dict[i])
    UpdateTree(stored_dict[i])
  return output

def AdaHuffDecode(bitstream):
  global count
  count=0
  NYT=Node('NYT',None,None,None,0)
  Decode_output=""
  index=0
  current=NYT
  while(index<len(bitstream)):
    symbol=None
    if current== NYT:                                        #Decoding the symbol that is a new symbol
        capture=bitstream[index:index+7]
        symbol=Convert_to_symbol(capture)
        Decode_output+=symbol
        setup_node=Node(symbol,None,None,None,1)
        restore_dict[symbol]=setup_node
        count+=1
        index=index+7
        if NYT.parent==None:
          ROOT_NODE=Node('ROOT',None,NYT,setup_node,1)
          NYT.parent=restore_dict[symbol].parent=ROOT_NODE
        else:
          parent=NYT.parent
          replace_node=Node('N'+str(count),parent,NYT,restore_dict[symbol],0)
          parent.lchild=NYT.parent=restore_dict[symbol].parent=replace_node
        current=ROOT_NODE
        
    elif current.lchild==None and current.rchild==None:    #Meaning we reached the symbol that is already added to the Tree
            symbol=current.tag
            Decode_output+=symbol
            current=ROOT_NODE          
                                               
    else:
          next_item=bitstream[index]
          if(next_item=='0'):
            current=current.lchild
          else:
            current=current.rchild
          index+=1
    if(symbol!=None):
      UpdateTree(restore_dict[symbol])
  return Decode_output

input_array=""
stored_dict={}                 #This will store all the symbols against their huffman codes
input=AdaHuffEncode(input_array)
restore_dict={}
AdaHuffDecode(input)

print(len(input)/len(input_array),'bpc')
