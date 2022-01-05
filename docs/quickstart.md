# Quickstart 

## Loading or creating transformations

### Linear or affine registrations (eg FSL FLIRT)

Registrations can be created from a `np.array`, a path to a text file that numpy can read, or by calling a wrapper for FLIRT. In all cases, the matrix should be 4x4 and the last row should be 0,0,0,1. 

```python 
import regtricks as rt 

# From an array
m = np.eye(4)
r = rt.Registration(m)

# From a file that numpy can read (NB if using a FLIRT matrix see below example)
p = '/a/path/to/file.txt'
r = rt.Registration(r)

# From a FLIRT matrix: provide the original source and reference images 
src = 'the_source.nii.gz'
ref = 'the_reference.nii.gz'
p = 'the_flirt_matrix.mat'
r = rt.Registration.from_flirt(p, src=src, ref=ref)

# Alternatively, you can run FLIRT directly and return a Reigstration object
src = 'the_source.nii.gz'
ref = 'the_reference.nii.gz'
r = rt.flirt(src, ref, **kwargs)
```


### Motion corrections (eg FSL MCFLIRT)

Motion corrections are stored as a sequence of Registrations (eg, for a timeseries of 50 volumes, there will be 50 registrations). They can be created from a list of `np.array`, a path to a text file that shaped (4xN) x 4, a path to a folder containing *only* files for the individual arrays, or by calling a wrapper for MCFLIRT. 

```python 
# From a list of arrays
m = [ np.eye(4) for _ in range(10) ] 
mc = rt.MotionCorrection(m)

# From a file that numpy can read, shaped (4xN) x 4
p = '/a/path/to/file.txt'
mc = rt.Registration(p)

# From a directory containing individual files, named in order
p = 'a/path/to/dir'
mc = rt.MotionCorrection(p)

# From a MCFLIRT -mats directory: provide the original src and ref images
# Unless using MCFLIRT's -reffile option, the src and the ref are the same!
src = 'the_source.nii.gz'
p = '/path/to/mcflirt.mat'
mc = rt.Registration.from_flirt(p, src=src, ref=src)

# Run MCFLIRT directly and return a MotionCorrection object
src = 'the_source.nii.gz'
mc = rt.mcflirt(src, **kwargs)
```

### Non-linear registrations (eg FSL FNIRT)



## Manipulating transformations 


## Applying transformations


## Working with ImageSpaces (voxel grids)