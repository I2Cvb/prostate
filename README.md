Prostate cancer research
========================

**Integration branch for Issue #6**

The goal is to segment design a caffe experiment where:
  - instad of volumes dataset, we use 2D slices dataset
  - select one patient for testing (actually a single slice of its volume where the tissues would be segmented)
  - use all the modalites (upsampled) and plug them at the first layer of the CNN-dl
  - 


Folder structure
----------------

```
.
|-- bin             # Folder where the C++ binary will be created
|-- build           # Folder that should be created to properly build the C++ tools
|-- cmake           # Folder containing the cmake information to properly compiled
|-- data		    # Folder containing the data
|-- doc			    # Folder containing the documentation
|-- notebook		# Folder containing notebooks with ideas
|-- pipeline		# Folder with the different pipeline
|-- publications	# Folder with the corresponding publications
|-- results		    # Folder with the results
|-- script		    # Folder containing the script which should call files of the pipeline
|-- src			    # Folder containing the source code developed for that application
|-- third-party		# Folder containing the third-party application
|-- LICENSE.md      # Licensing information
|-- README.md       # This file
`-- CMakeLists.txt  # File in order to build the C++ tools
````
