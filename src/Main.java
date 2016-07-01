
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.text.ParseException;

public class Main {

	static Configuration config;

	/**
	 * @param args
	 * @throws ParseException
	 * @throws IOException
	 */
	public static void main(String[] args) throws IOException, ParseException {
		config = new Configuration(args[0]);

		String[] trains = train();
		String model_base = trains[0];
		String train_config = trains[1];

		String[] tests = test(model_base);
		String test_base = tests[0];
		String test_config = tests[1];

		eval(model_base, test_base, train_config, test_config);
	}

	public static String[] train() throws IOException, ParseException {
		String train_visual_file = config.train_data_dir + File.separator
				+ "p_train_visual";
		String train_emotion_file = config.train_data_dir + File.separator
				+ "p_train_emotion";
		String train_text_file = config.train_data_dir + File.separator
				+ "p_train_text";

		VELDA velda = new VELDA();
		velda.initTrain(config.K, config.E, config.alpha_v,
				config.alpha_e, config.beta_v, config.beta_e, config.gamma_t,
				config.eta, train_visual_file, train_emotion_file,
				train_text_file, config.voc_v, config.voc_e, config.voc_t);
		String final_model_base = velda.train(config.model_dir,
				config.train_iters, config.save_intermediate_model);
		String train_config = velda.configure("train");
		return new String[] { final_model_base, train_config };
	}

	public static String[] test(String model_base) throws IOException {
		//String model_base = config.model_dir + File.separator
		//		+ config.train_iters;
		// Testing dataset
		String test_visual_file = config.test_data_dir + File.separator
				+ "p_test_visual";
		String test_emotion_file = config.test_data_dir + File.separator
				+ "p_test_emotion";
		String test_text_file = config.test_data_dir + File.separator
				+ "p_test_text";
		String p_zv_t_file = model_base + File.separator + "p_zv_t.txt";
		String p_ze_t_file = model_base + File.separator + "p_ze_t.txt";
		String p_zt_t_file = model_base + File.separator + "p_zt_t.txt";
		String p_t_r_file = model_base  + File.separator + "p_t_r.txt";

		VELDA velda = new VELDA();
		velda.initTest(p_zv_t_file, p_ze_t_file, p_zt_t_file, p_t_r_file,
				config.alpha_v, config.alpha_e, config.beta_v, config.beta_e,
				config.gamma_t, config.eta, test_visual_file, test_emotion_file,
				test_text_file);
		String test_base = velda.test(config.result_dir,
				config.test_iters, config.save_intermediate_model);
		String test_config = velda.configure("test");
		return new String[] { test_base, test_config };
	}

	public static void eval(String model_base, String test_base,
			String train_config, String test_config) throws IOException {
		String query_file = config.test_data_dir + File.separator
				+ "p_test_text";
		String visual_theta_file = test_base + File.separator + "p_d_zv.txt";
		String emotion_theta_file = test_base + File.separator + "p_d_ze.txt";
		String t_r_file = model_base + File.separator + "p_t_r.txt";
		String text_phi_file = model_base + File.separator + "p_zt_t.txt";

		String result_path = VELDAUtil.extractBase(test_base) + File.separator
				+ "result";
		
		File f = new File(result_path);
		if (!f.exists())
			f.mkdirs();

		PrintWriter pw = new PrintWriter(
				new FileWriter(result_path + File.separator + "log.txt"));
		pw.println(train_config);
		pw.println(test_config);
		pw.close();

		String result_file = result_path + File.separator + "result.csv";
		String top_results = result_path + File.separator + "top_results.csv";
		EvalUtil.retrieval(query_file, visual_theta_file,
				emotion_theta_file, t_r_file, text_phi_file, result_file,
				top_results);

	}

}