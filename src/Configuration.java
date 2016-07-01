import java.io.*;
import java.util.Properties;


public class Configuration {
	public int voc_v, voc_e, voc_t;
	public int K, E;
	public double alpha_v, alpha_e, beta_v, beta_e, gamma_t, eta;
	
	public int train_iters = 1000, test_iters = 100;
	public boolean save_intermediate_model;
	public String train_data_dir, test_data_dir, model_dir, result_dir; 
	
	public Configuration(String configFile) {
		Properties prop = new Properties();
		try {
			prop.load(new FileInputStream(configFile));
		} catch (FileNotFoundException e) {
			System.out.println("File not found:" + configFile);
			e.printStackTrace();
		} catch (IOException e) {
			System.out.println("Cannot load the file " + configFile);
		}
		
	
		this.voc_v = Integer.parseInt(prop.getProperty("voc_v"));
		this.voc_e = Integer.parseInt(prop.getProperty("voc_e"));
		this.voc_t = Integer.parseInt(prop.getProperty("voc_t"));
		
		alpha_v = Double.parseDouble(prop.getProperty("alpha_v")); // 1.0
		alpha_e = Double.parseDouble(prop.getProperty("alpha_e")); // 1.0
		beta_v = Double.parseDouble(prop.getProperty("beta_v")); // 0.1
		beta_e = Double.parseDouble(prop.getProperty("beta_e")); // 0.1
		gamma_t = Double.parseDouble(prop.getProperty("gamma_t")); // 0.1
		eta = Double.parseDouble(prop.getProperty("eta")); // 0.5
		
		K = Integer.parseInt(prop.getProperty("K"));
		E = Integer.parseInt(prop.getProperty("E"));
		
		train_iters = Integer.parseInt(prop.getProperty("train_iters"));
		test_iters = Integer.parseInt(prop.getProperty("test_iters"));
		save_intermediate_model = Boolean.valueOf(prop.getProperty("save_intermediate_model"));
		
		train_data_dir = prop.getProperty("train_data_dir");
		test_data_dir = prop.getProperty("test_data_dir");
		model_dir = prop.getProperty("model_dir");
		result_dir = prop.getProperty("result_dir");
		
	}

}
