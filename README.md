
# Intro to HDF5 and Python

This notebook shows my exploration of the HDF5 format for saving data. It is based by the book *Python and HDF5* by Andrew Collette.

The benefits of using a HDF5 fileformat is that it acts similar to a Numpy array, but the data lies on disk till it is requested. This makes it easy to deal with datasets so large that they do not fit in RAM.


```python
import h5py
import numpy as np
```

## Creating our first HDF5 datafile

We create and load HDF5 files by creating instances of the HDF5 File class.

To create an empty file (and overwrite any existing files with the same name) write the following.


```python
filename = 'our_first_datafile.hdf5'
with h5py.File(filename, 'w') as hdf_file:
    # Code manipulating the datafile
    pass
# The datafile is now closed
```

We have now created an empty HDF5 file in our current directory. The name of that file is *our_first_datafile.hdf5*. The *'w'* argument for this function symbolise that we want to create a new, empty file and overwrite any existing files.

There are several flags we can supply the HDF5 initialiser function, to do different things, they are

| Flag | Meaning                                                                       |
|:---- |:----------------------------------------------------------------------------- |
| 'w'  | Create new file and overwrite existing files                                  |
| 'w-' | Create new empty file but raise an error if it already exists                 |
| 'r'  | Open file in read only mode and raise error if the file doesn't exist         |
| 'r+' | Open file in read and write mode and raise error if the file doesn't exist    |
| 'a'  | Open file in read and write mode and create an empty file if it doesn't exist |

## Manipulating and reading the data in our datafile.

The benefit of HDF5 files is that they allow for a structured handling of data on disk, but in a way that is efficient for reading and writing. Data is stored in *datasets*, which is the HDF5 equivalent of a Numpy array.


```python
data = np.random.random((3, 2))
with h5py.File(filename, 'a') as hdf_file:
    hdf_file['dataset_1'] = data

print(data)
```

    [[0.09125845 0.09709407]
     [0.59468209 0.63913843]
     [0.62462814 0.05412021]]
    

---
What we just did was to create a dataset of the same shape and type as our numpy array, `data` and store the content of `data` in it. Note that if we try to overwrite an existing dataset this way, we will get an error message. If we want to change the content of a dataset, we have to do that through slicing. To change everything, we can write this.


```python
data = np.random.random((3, 2))
with h5py.File(filename, 'a') as hdf_file:
    hdf_file['dataset_1'][...] = data

print(data)
```

    [[0.99742839 0.66538952]
     [0.47259286 0.94203899]
     [0.89594797 0.59381092]]
    

---
We cannot directly look at the contents of a dataset, if we try to print it, we get this message.



```python
with h5py.File(filename, 'a') as hdf_file:
    dataset = hdf_file['dataset_1']
    print(dataset)

print(dataset)

```

    <HDF5 dataset "dataset_1": shape (3, 2), type "<f8">
    <Closed HDF5 dataset>
    

---
The reason for that is that the content of a datafile is only loaded to memory on-demand, so  you have to explicitly ask for the data you want to see. This can be done through a simple slicing. Here are some examples of this in action


```python
with h5py.File(filename, 'a') as hdf_file:
    some_data = hdf_file['dataset_1'][1:3, 1:2]
    all_data = hdf_file['dataset_1'][...]

print('Some data:')
print(some_data)
print('All data')
print(all_data)

print('Is the data loaded from the datafile the same as the data we last inserted?', np.array_equal(data, all_data))
```

    Some data:
    [[0.94203899]
     [0.59381092]]
    All data
    [[0.99742839 0.66538952]
     [0.47259286 0.94203899]
     [0.89594797 0.59381092]]
    Is the data loaded from the datafile the same as the data we last inserted? True
    

---
There are several ways to insert datasets in a datafile. For example, if we know that we want to insert a 10000x10000 matrix in the datafile, then we tell the HDF5 file that we will insert that much data, but the storage space will be allocated as we insert data.

We can also control what type the dataset should take (say 16 bit float instead of 64 bit). The type casting will then be performed automatically by the HDF5 software while it is writing the data to disk. 

A better way of creating new datasets is through the create_dataset function, which gives us more control over the structure of the dataset. Below is an example of this in action.



```python
with h5py.File(filename, 'a') as hdf_file:
    hdf_file.create_dataset('empty_dataset1', shape=(3, 1))
    hdf_file.create_dataset('empty_dataset2', shape=(10, 4), dtype=np.float16)
    hdf_file.create_dataset('ones_dataset1', data=np.ones((3, 3), dtype=np.complex64))
    
    dataset1 = hdf_file['empty_dataset1'][...]
    dataset2 = hdf_file['empty_dataset2'][...]
    dataset3 = hdf_file['ones_dataset1'][...]
    
print(dataset1)
print(dataset2)
print(dataset3)
```

    [[0.]
     [0.]
     [0.]]
    [[0. 0. 0. 0.]
     [0. 0. 0. 0.]
     [0. 0. 0. 0.]
     [0. 0. 0. 0.]
     [0. 0. 0. 0.]
     [0. 0. 0. 0.]
     [0. 0. 0. 0.]
     [0. 0. 0. 0.]
     [0. 0. 0. 0.]
     [0. 0. 0. 0.]]
    [[1.+0.j 1.+0.j 1.+0.j]
     [1.+0.j 1.+0.j 1.+0.j]
     [1.+0.j 1.+0.j 1.+0.j]]
    

---
Note that HDF5 will not warn before overflow errors during type-casting. To demonstrate this, see the example below.



```python
with h5py.File(filename, 'a') as hdf_file:
    hdf_file['empty_dataset1'][1] = 2.**65
    print(hdf_file['empty_dataset1'][1])
    
    hdf_file['empty_dataset2'][1, 1] = 2.**65
    print(hdf_file['empty_dataset2'][1, 1])
```

    [3.689349e+19]
    inf
    

---
Notice how the value of the float16 dataset simply was set to infinity? This is a *gotcha* to be aware of when using limited capacity numericals in HDF5 files.

We have now created a datafile and demonstrated how values are typecasted to the appropriate type when they are written into a datafile. Now we want to do this typecasting at read-time. There are several ways to do this, and here are two.


```python
with h5py.File(filename, 'a') as hdf_file:
    dataset = hdf_file['empty_dataset1']
    
    # Method 1
    out1 = np.empty(dataset.shape, dtype=np.float16)
    dataset.read_direct(out1)
    
    # Method 2
    with dataset.astype('float16'):
        out2 = dataset[...]
        
print(out1.dtype)
print(out1)
print(out2.dtype)
print(out2)
```

    float16
    [[ 0.]
     [inf]
     [ 0.]]
    float16
    [[ 0.]
     [inf]
     [ 0.]]
    

---
Notice here, that once again the value(s) outside the numerical range of the variable we cast to are set to infinity without warning.

---

## Slicing

An integral part of the python framework is slicing. Both Numpy and Pandas have powerful slicing capabilities, and so does HDF5. 


```python
with h5py.File(filename, 'a') as hdf_file:
    dataset = hdf_file.create_dataset('random', data=np.random.randint(0, 10, (5, 10)))
    
    # Get all data
    data = dataset[...] # [...] is the same as [:, :]
    
    # Get all data in the second row
    some_row = dataset[1:2, :]
    
    # Get all data in the third column
    some_column = dataset[:, 2:3]
    
    # Extract a matrix starting at the first row, ending at the third
    # and starting at the second column, ending at the fifth
    some_matrix = dataset[1:5, 0:3]
    

    
print('All data:')
print(data)
print('A row of data:')
print(some_row)
print('A column of data:')
print(some_column)
print('A matrix of data:')
print(some_matrix)
```

    All data:
    [[1 9 5 3 0 3 6 0 0 6]
     [7 8 3 8 1 4 1 1 7 8]
     [6 4 0 8 3 0 1 5 1 1]
     [2 6 3 8 8 8 6 9 2 1]
     [2 1 7 1 2 3 1 2 6 2]]
    A row of data:
    [[7 8 3 8 1 4 1 1 7 8]]
    A column of data:
    [[5]
     [3]
     [0]
     [3]
     [7]]
    A matrix of data:
    [[7 8 3]
     [6 4 0]
     [2 6 3]
     [2 1 7]]
    

---
It is important to note that boolean slicing is accepted, however, we cannot perform them as simply as with a numpy array. The following code shows this by raising an exception.


```python
with h5py.File(filename) as hdf_file:
    dataset = hdf_file['random']
    
    # Try a simple bool-slicing to get numbers greater than 5
    gt5 = dataset[dataset > 5]
    
print('The numbers in our random dataset that are greater than five are')
print(gt5)
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    <ipython-input-11-46cda19b51bf> in <module>
          3 
          4     # Try a simple bool-slicing to get numbers greater than 5
    ----> 5     gt5 = dataset[dataset > 5]
          6 
          7 print('The numbers in our random dataset that are greater than five are')
    

    TypeError: '>' not supported between instances of 'Dataset' and 'int'


---
If you want to use boolean slicing, you need to directly supply the boolean mask that you want to use, as the HDF5 dataset does not allow for direct mathematical operations like Numpy arrays do.

---

### Note, optimizing the read speed.

We have here used slicing directly to get the data, instead of first allocating memory using ```np.empty``` and then filling it with ```dataset.read_direct```. The direct slicing notation is more readable, but the read_direct command is faster. There are some internal HDF5 reasons for this, but I did not bother to spend time understanding why. If we want to slice using this approach, we can do that the following way.


```python
with h5py.File(filename) as hdf_file:
    dataset = hdf_file['random']
    data = np.empty((2, dataset.shape[1]))
    
    dataset.read_direct(data, np.s_[1:5:2, :])

print(data)
```

    [[7. 8. 3. 8. 1. 4. 1. 1. 7. 8.]
     [2. 6. 3. 8. 8. 8. 6. 9. 2. 1.]]
    

---

## Resizing datasets
One important feature of any datasaving system is the option to extend it easily. HDF5 supports this, however it requires that this was enabled when the dataset was created. Here we demonstrate that it fails.


```python
with h5py.File(filename) as hdf_file:
    dataset = hdf_file['random']
    dataset.resize((np.prod(dataset.shape), 1))
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    <ipython-input-13-d3104c67c171> in <module>
          1 with h5py.File(filename) as hdf_file:
          2     dataset = hdf_file['random']
    ----> 3     dataset.resize((np.prod(dataset.shape), 1))
    

    C:\NMBU\Miniconda3\lib\site-packages\h5py\_hl\dataset.py in resize(self, size, axis)
        345         with phil:
        346             if self.chunks is None:
    --> 347                 raise TypeError("Only chunked datasets can be resized")
        348 
        349             if axis is not None:
    

    TypeError: Only chunked datasets can be resized


---
Let us now create a new dataset which we can reshape.


```python
with h5py.File(filename) as hdf_file:
    dataset = hdf_file.create_dataset(name='reshapable', shape=(2, 2), 
                                      dtype='float32', maxshape=(None, None))
    print(dataset[...])
    dataset.resize((4, 1))
    print(dataset[...])
```

    [[0. 0.]
     [0. 0.]]
    [[0.]
     [0.]
     [0.]
     [0.]]
    

---
The ```maxshape``` argument is needed if one want a dataset to be resizable. It (obviously) specifies the maximum size of the dataset. If we, however, do not know the maximum length of an axis, we can write None. This simply means that there is no maximum length of that axis.

### Be careful when reshaping.

In Numpy, reshaping arrays will shuffle the elements around, as demonstrated below


```python
a = np.array([1, 2, 3, 4])
a = a.reshape((2, 2))
print(a)
```

    [[1 2]
     [3 4]]
    

However, when we reshape a dataset, this is not the behaviour we get. as demonstrated below.


```python
with h5py.File(filename) as hdf_file:
    dataset = hdf_file['reshapable']
    dataset[...] = np.arange(2, 6).reshape((4, 1))
    print('Before reshaping\n', dataset[...])
    dataset.resize((2, 2))
    print('After reshaping \n', dataset[...])
```

    Before reshaping
     [[2.]
     [3.]
     [4.]
     [5.]]
    After reshaping 
     [[2. 0.]
     [3. 0.]]
    

---
It is clear that one should be careful when reshaping. Note that the first two values were kept. this is all a result of hw HDF5 internally deals with resizing datasets. Luckily, this is not an issue when extending datasets, as demonstrated below.


```python
with h5py.File(filename) as hdf_file:
    dataset = hdf_file['reshapable']
    dataset[...] = np.arange(2, 6).reshape((2, 2))
    print('Before adding row\n', dataset[...])
    dataset.resize((2, 3))
    print('After adding row \n', dataset[...])
    print('--------------------------------')
    print('Filling empty row')
    print('--------------------------------')
    dataset[...] = np.arange(2, 8).reshape((2, 3))
    print('Before adding column\n', dataset[...])
    dataset.resize((3, 3))
    print('After adding column\n', dataset[...])
```

    Before adding row
     [[2. 3.]
     [4. 5.]]
    After adding row 
     [[2. 3. 0.]
     [4. 5. 0.]]
    --------------------------------
    Filling empty row
    --------------------------------
    Before adding column
     [[2. 3. 4.]
     [5. 6. 7.]]
    After adding column
     [[2. 3. 4.]
     [5. 6. 7.]
     [0. 0. 0.]]
    

---

# Grouping

One integral part of the HDF file format is grouping. It can be considered as a folder structure within the datafile. A group might contain several subgroups and datasets. This gives a very orderly way of dealing with data. Below is a simple example where we create a group and a dataset within it.


```python
with h5py.File(filename) as hdf_file:
    group = hdf_file.create_group('some_group')
    dataset1 = group.create_dataset(name='first_ordered_dataset', shape=(3, 3),
                                    dtype='float32')
    group['second_ordered_dataset'] = np.ones((2, 2))
```

If we want to view the data in one of the datasets, we can do it these ways.


```python
with h5py.File(filename) as hdf_file:
    # First way
    data1 = hdf_file['some_group/first_ordered_dataset'][...]
    
    # Second way
    group = hdf_file['some_group']
    data2 = group['second_ordered_dataset'][...]

print('First data \n', data1)
print('Second data \n', data2)
```

    First data 
     [[0. 0. 0.]
     [0. 0. 0.]
     [0. 0. 0.]]
    Second data 
     [[1. 1.]
     [1. 1.]]
    

---
We can also create subgroups. Below are some examples where we do that.


```python
with h5py.File(filename) as hdf_file:
    # Creating a single subgroup in existing group
    subgroup1 = hdf_file['some_group'].create_group('some_subgroup')
    
    # Creating a several nested subgroups at once
    subgroup2 = hdf_file.create_group('a_group/a_subgroup/a_subsubgroup')
```

In the above code, the line ```subgroup2 = hdf_file.create_group('a_group/a_subgroup/a_subsubgroup')``` first created the group ```a_group```, it then created a subgroup named ```a_subgroup``` within ```a_group``` before it finally created the subgroup ```a_subsubgroup``` within ```a_subgroup```. The variable ```subgroup2``` became the innermost group (```a_subsubgroup```).

We can also create groups at the same time as we create a dataset. The code below demonstrate this.


```python
with h5py.File(filename) as hdf_file:
    # Creating a single subgroup in existing group
    subgroup1 = hdf_file.create_dataset(name='new_group/new_dataset', shape=(3, 3),
                                        dtype='int')
```

If we want to view the content of this dataset, we can simply write the following


```python
with h5py.File(filename) as hdf_file:
    data = hdf_file['new_group/new_dataset'][...]

print(data)
```

    [[0 0 0]
     [0 0 0]
     [0 0 0]]
    

---

## Iterating over a datafile
We have now almost reached the end of the notes. Before we head on to the final chapter, we quickly go through how we iterate over the content of a HDF5 file. Iterating over a HDF5 file in Python works similarly to how we iterate over dictionaries, as demonstrated below.


```python
with h5py.File(filename) as hdf_file:
    for i in hdf_file:
        print(i)
```

    a_group
    dataset_1
    empty_dataset1
    empty_dataset2
    new_group
    ones_dataset1
    random
    reshapable
    some_group
    

---
As you see, we iterate over the entries that we can use when iterating over a HDF5 file. If we want to iterate over the groups within a HDF5 file, we can do that this way


```python
with h5py.File(filename) as hdf_file:
    for i in hdf_file:
        if isinstance(hdf_file[i], h5py.Dataset):
            continue  # Don't show dataset names
        print(i)
```

    a_group
    new_group
    some_group
    

---
We can also iterate over the entries of a group the exact same way as we iterate over the entries of a File (in fact, the File class is a subclass of the Group class).


```python
with h5py.File(filename) as hdf_file:
    for i in hdf_file['some_group']:
        if isinstance(hdf_file['some_group'][i], h5py.Dataset):
            continue  # Don't show dataset names
        print(i)
```

    some_subgroup
    

## Chunking - A method to optimize read-time

It is helpful to imagine a HDD (or SSD) and RAM as strips of paper where numbers representing our data is written one after another. When we want to read a file, it is quicker if the data is stored physically close inside the computer, instead of spread out. Below is a figure that demonstrate how data can be stored as a contiguous block or as several fragments.

![Figure showing data stored as a contiguous chunk and data stored as several fragments](https://notebooks.azure.com/yngvemoe/libraries/masternotebooks/raw/HDF5%2Ffigures%2Fdata_on_disk.png)

Here we come to an interesting problem, images are two dimensional constructs, add channels and they become three dimensional. The moment we have a series of colour images, we have a four dimensional construct that we want to store as a one dimensional list. To do this, we obviously have to "unwrap" the data, and the way we do that can severly impare read speed.

Imagine, for example, if the red-value of the top-left pixel of the first image is stored next to the red-value of the top-left pixel of the second image. This will make it fast to load the red-values of all the top-left pixels at the same time. However, loading a full image will take time, as the pixel just below the top-left pixel is separated from the value of the top-left pixel.

Luckily, HDF5 allows us to manually say how the pixels should be stored. This is called chunking. I will not elaborate much here, just enough to get an efficient pipeline to load 2D images. For an excellent introduction to this topic (and compression of HDF5 files) I reccomend reading the book *Python and HDF5* by Andrew Collette.

---
Data in HDF5 is by default stored so that the values of the last axis is physically adjacent to each other. The line ```data = dataset[1, 1, :]``` will take shorter time than the line ```data = dataset[1, :, 1]```, which again will take shorter time than the line ```data = dataset[:, 1, 1]```.

Imagine now that we have images stored on the format \[image_number, height, width, channel\]. This will make it fast to load the colour values of a single pixel, which might be exactly what we want. However, we might also be interested in it being faster to get the colour values of a single channel in an image, rather than to get all the colours of a single pixel. To do this, we create the dataset in the following way.


```python
with h5py.File(filename) as hdf_file:
    dataset = hdf_file.create_dataset(name='chunking', shape=(4, 10, 10, 3), 
                                      maxshape=(None, 10, 10, 3), 
                                      chunks=(1, 10, 10, 1))
    
```

What we did here was to create a dataset called *chunking*, consisting of 4 10-by-10 RGB images. We specified that we can add new images, but not change the shape or number of colour bands of the images. We also specified that we want the second and third stored as close as possible on disk.

Making sure that data is stored this way is significantly more important when data is stored on a HDD rather than on an SSD. This is because on a HDD, the pin reading the disk has to physically move a lot when reading fragmented data. This is not the case in a solid state drive.

## Concluding notes
There are other important aspects of the HDF5 file type such as how to attach metadata and attributes to datasets and groups. However, those will not be covered in these notes for now (but might be added later). 

If these notes caught your attention, I very much reccomend buying the book *Python and HDF5* it is a clearly written book, with easy to understand examples. It also goes further into implementation details but not further than anyone that simply use HDF5 as a tool in their data pipeline needs to understand.
