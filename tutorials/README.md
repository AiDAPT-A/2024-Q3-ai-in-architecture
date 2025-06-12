Python tutorials are designed according to the following principles:

- *No-experience*. Students with no or limited background should be able to relatively easily follow it,
- *Visually-engaging*. The tutorials are highly visual to make it easier to grasp concepts and ideas, 
- *Context-driven*. The tutorials are designed around an architectural application / need / question - and *not* around a topic in programming,
- *Incremental complexity*. The tutorials are ordered from simple to more complex, and build on each other. Later tutorials often require the programming skills learned in earlier ones.

The tutorials are *application-driven* because the tutorials are meant for students in architectural design. 
Therefore, to maximally engage this audience group, a tutorial starts with an architectural application (or need, or question). 
The necessary programming concepts (and associated Pyhon libraries) that are needed to help 'solve' the application are woven into the tutorials, and are explained and used only when needed. 
Hence, instead of simply teaching a collection of programming concepts, the tutorials directly associate programming concepts to particular architectural applications. 

| Tutorial | Learning objectives |
| --- | --- |
| [T0_Intro_to_Python_and_Colab](./T0_Intro_to_Python_and_Colab.ipynb) | Get familiar Python programming in Google Colab|
| [T1_From_Code_to_Canvas](./T1_From_Code_to_Canvas.ipynb) | Use Google Collab and run code <br>Create and print most common data types <br>Create and manipulate polygonal shapes <br>Plot polygonal shapes <br>Use for loops and functions|
| [T2_From_Numbers_to_Plots](./T2_From_Numbers_to_Plots.ipynb) | Make use of CSV files to create a DataFrame <br>Cleaning Data (reading, sorting, and selecting) <br>Plotting FloorPlans|
| [T3_From_Geometries_to_Graphs](./T3_From_Geometries_to_Graphs.ipynb) | Define a graph <br>Create, manipulate, and visualize a graph in Python <br>Describe the access graph of a floor plan <br>Extract (apartment-level) access graphs from the IFC building elements.|
| [T4_From_Footprints_to_Photos](./T4_From_Footprints_to_Photos.ipynb) | Visualize and interpret *building+context* representations <br>Automatically collect aerial images and create a customized dataset <br>Locate building footprints from geographical information|
| [T5_From_Photos_to_Embeddings](./T5_From_Photos_to_Embeddings.ipynb) | Generate image embeddings from pre-trained foundation models <br>Compute the cosine similarity between embeddings <br>Interpret *building+context* representations|
| [T6_From_Images_to_3D_Understanding](./T6_From_Images_to_3D_Understanding.ipynb) | Understand opportunities of AI in computer vision and photogrammetry <br>Introduce the concept of data fusion when working with multiple modalities|
| [T7_From_Graphs_to_Similarity](./T7_From_Graphs_to_Similarity.ipynb) | Investigate floor layout similarity using pre-trained deep neural networks specialized in extracting layout-specific features|
| [T8_Similarity_Urban_Scale](./T8_Similarity_Urban_Scale.ipynb) | Visualize and interpret *urban* representations at scale <br>Create a customized dataset of aerial and street view images <br>Generate an urban similarity score by combining similarity scores computed from both aerial and street view images|

