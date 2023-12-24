Neural Network Java Implementation
Ke Xu - kxu27@u.rochester.edu

Note: When building the neural network, 3Blue1Brown's video helped me greatly: https://www.youtube.com/watch?v=Ilg3gGewQ5U&ab_channel=3Blue1Brown. I was struggling with why the accuracy of the iris dataset is 0.33 with one lay of 3 nodes. As I cranked up the layer size from 3 to 7 to 100, the accuracy improved significantly. Hence I made one layer of 100 nodes as default for the iris dataSet. (I might over-complicate the model). When searching for the MNIST, I found a CSV version that provided a label first and input second to create a simple reading function (https://www.kaggle.com/datasets/oddrationale/mnist-in-csv). Since this set is different from the iris, I have to create different data reader, transformation, and file writer (essentially a different train function. Why? Because I can train the method with randomized smaller samples and write files during each step but not record data in a Fibonacci run time).


When running the main, the system provides a quick sample print of the nn and writes the accuracy data into the file: nn_graph_data.txt. Hence, I can take those data into Excel and provide a report graph.


There are some assumed default attributes for nn:
	For the neural network, the iris dataset has a layer of 100, and MNIST has a layer of 300 and 800. Iris can provide a sample overview, but MNIST only focuses on providing the graph. To optimize run time and get an efficient overview for MNIST, I implemented random sampling of training and testing sets. The size of the training and testing sets are 5000 and 2500. The writing accuracy increment is 2 by default to minimize run time. The learning rate is assumed decay rate. 

Those are the attributes that can be edited in those files.

To build the main class, you can recompile the class by directory in the folder: 

	javac javac nn/*.java math/util/*.java Main.java

To Run nn: java nn/Main.java [Mode] [epochs/steps]

	Mode: 0 -> iris data set, 1 -> MNIST data set


There are a lot of things to improve on: I would create a reading and outputting data interface with methods so I can apply them to different data sets. I would also improve the bias since no additional weight is attached to each layer. I should also create training set divisions and introduce generations.


--------------------------------------------------

SAMPLE OUTPUT:

jackxu@XuWOWdeMBP src % java nn/Main.java 0 200 
Data Set 	iris.data.txt
Activation of A Random Example: 
6.3 3.3 6.0 2.5 
0.7344972934781058 2.068636916699776E-6 0.9999999999701108 7.52755700921928E-4 2.236023538065685E-14 4.5767584067234466E-9 4.958920692854172E-6 0.017788937954935812 0.005350532022683536 2.7743831322858415E-10 0.9993044590060164 0.9999343434744061 0.22725813030275252 0.059081752229763015 0.9929533992379717 0.99999756077026 1.944040376033749E-12 0.17093556027868123 0.9999999999984011 1.2301489727430964E-9 0.9999999999999865 0.3163232375139333 0.99707426609894 6.264993849542105E-4 1.322357770057802E-4 3.854554106726871E-4 0.017948837307787867 0.016201497030581588 3.120224449500487E-9 0.8648188221829064 2.0844813491296876E-10 3.173175079873714E-9 1.0938344454333441E-7 0.9999999997952225 0.9999735068551995 3.2456915485566464E-7 5.031631349047657E-12 0.9946283727596266 0.9999999368868391 1.226413500612009E-12 0.998783905129095 2.1363215868607935E-8 0.9999999995985251 0.999999999999998 0.6633472835412662 1.7664409602443647E-13 3.1609710701976368E-12 1.612350631095144E-11 0.9999999999997278 3.977353431749057E-11 0.9872766299660206 0.9999999268236699 0.028401167295303183 0.9999999849187764 0.9999840665226253 0.9999998701217649 0.9999999998165734 0.973782217819811 0.9999999999999636 9.931393424826904E-9 0.9844967001027259 1.2026060847284567E-10 0.9999999986877395 0.9999999968401776 0.9500283086934935 0.9998946903095071 3.9067697767407046E-14 0.6172799721907483 0.9999999999999989 0.9940007884703408 0.4344511134680698 5.091728953273238E-6 0.9994925245567147 0.7354512917449111 9.717502592292832E-9 0.9999999999489777 0.9969835179882589 0.22463982802922694 2.0789701076689235E-6 0.9999999998549163 0.7182136862594021 0.9974663424201647 0.9999999999999098 0.9999999999999045 7.565331076482899E-14 0.9999999994951432 0.9999999963099611 0.09648437591790307 0.9999208083032518 0.9831867626075911 0.9999918614984226 7.405148952583256E-6 0.9997090988946468 0.999999999997023 8.876591720230482E-5 0.9999999996331017 0.2252159399801783 7.799460855158901E-23 1.539252170852005E-6 0.9996090974177411 
0.33876271583603695 0.9052168542739863 0.004112380149189628 
Accuracy percentage: 0.8533333333333334
Data has been written to the file successfully.

-----------------------------------------------
