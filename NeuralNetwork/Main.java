import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Set;
import java.util.HashSet;
import java.util.Hashtable;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import nn.*;
/*
 * @author: Ke Xu
 * Kxu27@u.rochester.edu
 */


public class Main {
	static double count = 0.0;
	static int inputSize;
	static int outputSize;
	static Hashtable<String, Double> outputTable = new Hashtable<>();
	static String filename = "examples/";
	//increment of change
	static int increment = 2;
	
	public static void main(String[] args) {
		
		//to edit
		Path filePath = Path.of("nn_graph_data.txt");
		boolean buildingFile = true;
		
		
		int mode = Integer.parseInt(args[0]);
		int step = Integer.parseInt(args[1]);
		String input;
		if(mode == 0) {
			input = "iris.data.txt";
		} else {
			input = "mnist_train.csv";
		}
		
		

		
		try {
			double[][] data;
			double[][] testingData;
			ArrayList<Example> examples = new ArrayList<>();
			ArrayList<Example> examplesTest = new ArrayList<>();
			
			//for iris data set
			if(input.equals("iris.data.txt")) {
				data = readData1(filename+input);
				
				//convert into example
				for(double[] row: data) {
		    		double[] input_temp = new double[row.length - 1];
		    		double[] output_temp = new double[(int)count];
		    		Arrays.fill(output_temp, 0.0);
		        	System.arraycopy(row, 0, input_temp, 0, row.length - 1);
		        	output_temp[(int)row[row.length - 1]] = 1.0;
		        	
		        	examples.add(new Example(input_temp, output_temp));
		    	}
				
				//sample output
				NeuralNetwork nn = new NeuralNetwork(examples.get(0).inputs.length,100,examples.get(0).output.length);
				nn.train(examples, 100, examples.size(), new DecayingLearningRateSchedule());
				
				double[][] act = nn.getActivations();
				
				System.out.println("Data Set \t"+input);
				System.out.println("Activation of A Random Example: ");
				for(double[] r: act) {
					for(double val: r) {
						System.out.print(val + " ");
					}
					System.out.println();
				}
				
				System.out.println("Accuracy percentage: " + nn.accuracy(examples));
				
//				System.out.println();
//				for(Layer l: nn.getHiddenLayers()) {
//					System.out.print(l.toString());
//				}
				
				//writing file
				if(buildingFile) {
					//increment by 4
		        	write_accuracy(filePath, step, examples);
		    	}
				
			//for mnist_train.csv
			} else {
				System.out.println("Data Set \t"+input);
				data = readData2(filename+input);
				
				for(double[] row: data) {
		    		double[] input_temp = new double[row.length - 1];
		    		double[] output_temp = new double[10];
		    		Arrays.fill(output_temp, 0.0);
		        	System.arraycopy(row, 1, input_temp, 0, row.length - 1);
		        	output_temp[(int)row[0]] = 1.0;
		        	
		        	examples.add(new Example(input_temp, output_temp));    
		        
				}
				
				testingData = readData2("src/examples/" + "mnist_test.csv");
				
				for(double[] row2:  testingData) {
		    		double[] input_temp = new double[row2.length - 1];
		    		double[] output_temp = new double[10];
		    		Arrays.fill(output_temp, 0.0);
		        	System.arraycopy(row2, 1, input_temp, 0, row2.length - 1);
		        	output_temp[(int)row2[0]] = 1.0;
		        	
		        	examplesTest.add(new Example(input_temp, output_temp));
				}
				
//				for(int i = 0; i < 5; i++) {
//				System.out.println(examples.get(i).toString());
//				}	
				
				if(buildingFile) {
		        	write_accuracy2(filePath, step, examples, examplesTest);
		    	}
			}
				
		} catch (IOException e) {
	        e.printStackTrace();
		}
	}
	
	//for writing iris dataSet graphing, find the trend between accuracy and training set size
	public static void write_accuracy(Path filePath, int step, ArrayList<Example> examples) {
		
		 ArrayList<String> lines = new ArrayList<>();
		 
		 //for testing training set size
//		for(int i = increment; i < examples.size(); i += increment) {
//			NeuralNetwork nn = new NeuralNetwork(examples.get(0).inputs.length,100,examples.get(0).output.length);
//			nn.train(examples, 100, i, new DecayingLearningRateSchedule());
//			String line = Integer.toString(i) + "," + Double.toString(nn.accuracy(examples));
//			lines.add(line);
//		}
		 
		 for(int i = 1; i < step; i += increment) {
				NeuralNetwork nn = new NeuralNetwork(examples.get(0).inputs.length,100,examples.get(0).output.length);
				nn.train(examples, i, examples.size(), new DecayingLearningRateSchedule());
				String line = Integer.toString(i) + "," + Double.toString(nn.accuracy(examples));
				lines.add(line);
			}
		
		try {
			try {
        		Files.deleteIfExists(filePath);
   	 		} catch (IOException e) {
        		e.printStackTrace();
   	 		}
            // Write the lines to the file
            Files.write(filePath, lines, StandardOpenOption.CREATE, StandardOpenOption.WRITE);

            System.out.println("Data has been written to the file successfully.");
        } catch (IOException e) {
            System.err.println("Error writing to the file: " + e.getMessage());
        }
	}
	
	//for writing mnist dataSet graphing, find the trend between epochs and number of errors
	public static void write_accuracy2(Path filePath, int steps, ArrayList<Example> examples, ArrayList<Example> testing) {
		
		
			 NeuralNetwork nn = new NeuralNetwork(examples.get(0).inputs.length,300,800,examples.get(0).output.length);
			 nn.train2(examples, testing, steps, new DecayingLearningRateSchedule());

		
		try {
			try {
        		Files.deleteIfExists(filePath);
   	 		} catch (IOException e) {
        		e.printStackTrace();
   	 		}
           // Write the lines to the file
           Files.write(filePath, nn.lines, StandardOpenOption.CREATE, StandardOpenOption.WRITE);

           System.out.println("Data has been written to the file successfully.");
       } catch (IOException e) {
           System.err.println("Error writing to the file: " + e.getMessage());
       }
	}
	
	//for reading iris dataset
	public static double[][] readData1(String filePath) throws IOException {
    	Set<String> outputSet = new HashSet<>();
    	
    	BufferedReader br = null;
    	
        try {
        	br = new BufferedReader(new FileReader(filePath));
            String line;
            int numLines = 0;

            // Count the number of lines in the file
            while ((line = br.readLine()) != null) {
            	
            	if (line.trim().isEmpty()) {
                    continue;
                }
            	
            	String[] values = line.split(",");
            	//add output value
            	if(!values[values.length-1].trim().equals("")) {
                    outputSet.add(values[values.length-1]);
            	}
                numLines++;
            }
            
            //make table
            for (String element :outputSet) {
            	outputTable.put(element, count);
            	count++;
            }
            
            // Initialize the 2D array to store the data
            double[][] data = new double[numLines][];
            
            // Reset the reader
            br.close();
            br = new BufferedReader(new FileReader(filePath));

            // Read data from the file and store it in the array
            int lineIndex = 0;
            while ((line = br.readLine()) != null) {
            	
            	if (line.trim().isEmpty()) {
                    continue;
                }
            	
                String[] values = line.split(",");
                
                double[] row = new double[values.length];

                // Parse each value and store it in the array
                for (int i = 0; i < values.length; i++) {
                	if(i == values.length - 1) {
                		//convert output label into number
                		row[i] = outputTable.get(values[i]);
                	} else {
                        row[i] = Double.parseDouble(values[i]);
                	}
                }
                data[lineIndex++] = row;
            }

            return data;
        } finally {
        	if (br != null) {
                try {
                    br.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }	
        }
    }
	
	//for reading minist dataset
	public static double[][] readData2(String filePath) throws IOException {    	
    	BufferedReader br = null;
    	
        try {
        	br = new BufferedReader(new FileReader(filePath));
            String line;
            int numLines = 0;
            boolean isFirstLine = true;
            // Count the number of lines in the file
            while ((line = br.readLine()) != null) {
            	if (isFirstLine) {
                    isFirstLine = false;
                    continue; // Skip the first line
                }
            	if (line.trim().isEmpty()) {
                    continue;
                }
                numLines++;
            }
            
           
            // Initialize the 2D array to store the data
            double[][] data = new double[numLines][];
            
            // Reset the reader
            br.close();
            br = new BufferedReader(new FileReader(filePath));

            // Read data from the file and store it in the array
            int lineIndex = 0;
            isFirstLine = true;
            while ((line = br.readLine()) != null) {
            	
            	if (line.trim().isEmpty()) {
                    continue;
                }
            	
            	if (isFirstLine) {
                    isFirstLine = false;
                    continue; // Skip the first line
                }
            	
                String[] values = line.split(",");
                
                double[] row = new double[values.length];

                // Parse each value and store it in the array
                for (int i = 0; i < values.length; i++) {
                    row[i] = Double.parseDouble(values[i]);
                }
                data[lineIndex++] = row;
            }

            return data;
        } finally {
        	if (br != null) {
                try {
                    br.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }	
        }
    }
}
