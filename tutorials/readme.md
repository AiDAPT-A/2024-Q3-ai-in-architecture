Python tutorials are designed according to the following principles:

- *No-experience*. Students with no or limited background should be able to relatively easily follow it,
- *Visually-engaging*. The tutorials are highly visual to make it easier to grasp concepts and ideas, 
- *Context-driven*. The tutorials are designed around an architectural application / need / question - and *not* around a topic in programming,
- *Incremental complexity*. The tutorials are ordered from simple to more complex, and build on each other. Later tutorials often require the programming skills learned in earlier ones.

The tutorials are *application-driven* because the tutorials are meant for students in architectural design. 
Therefore, to maximally engage this audience group, a tutorial starts with an architectural application (or need, or question). 
The necessary programming concepts (and associated Pyhon libraries) that are needed to help 'solve' the application are woven into the tutorials, and are explained and used only when needed. 
Hence, instead of simply teaching a collection of programming concepts, the tutorials directly associate programming concepts to particular architectural applications. 

The tutorials are organized as follows:
Visualize and interpret building+context representations
Automatically extract aerial images and create a customized dataset
Generate image embbedings from pre-trained foundation models
Locate building footprints from geographical information
Compute the distance between feature vectors

| Tutorial | Learning objectives |
| --- | --- |
| [Code canvas](./1_code_canvas.ipynb) | Create an understand how python objects can effectively represent shapes <br>Develop functions that output shapes and can modify them <br>Learn to write a script that can automatically develop a floor layout|
| [3D to 2D](./2_from_3D_to_2D.ipynb) | Mounting Google Drive in Google Colaboratory <br>Creating GeoPandas DataFrame <br>Cleaning Data <br>Plotting FloorPlans|
| [Building blueprint](3_building_blueprint.ipynb) | Describe a floor layout as a composed shape, an image, and an access graph <br> Effectively represent a composed shape, an image, and an access graph in Python objects <br> Create rasterized images from the composed shape representation <br> Extract access graph from the composed shape representation |
| Street view imagery |  |
| [Beyond boundaries](4_beyond_boundaries.ipynb) | Visualize and interpret building+context representations <br> Automatically extract aerial images and create a customized dataset <br>Generate image embeddings from pre-trained foundation models <br> Locate building footprints from geographical information <br> Compute the distance between feature vectors |
| ... |  |
| Learning from buildings |  |
| Architectural harmony |  |
