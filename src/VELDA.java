import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.text.ParseException;
import java.util.ArrayList;
import java.util.Arrays;
import java.lang.Math;

public class VELDA {

	// ---------------------------------------------------------------
	// Model Parameters and Variables
	// ---------------------------------------------------------------
	public int K, E, R=2; // # of visual, emotional topics and relevance type
	public double alpha_v, alpha_e, beta_v, beta_e, gamma_t, eta; // hyperparameters
	public double Vbeta, Ebeta, Kalpha, Ealpha, Tgamma; // Intermediate
														// variables

	public String visualFile, textFile, emotionFile; // input data file
	public ArrayList<int[]> visual_corpus, text_corpus, emotion_corpus;

	public int D; // dataset size (i.e., number of docs)
	public int voc_v; // vocabulary size for visual words
	public int voc_e; // vocabulary size for emotional words
	public int voc_t; // vocabulary size for textual words

	// topic assignments for visual/emotional words in each doc,
	// size = M * D
	public ArrayList<int[]> zv_d_n, ze_d_n, zt_d_n;

	// n_d_zv, # of visual words in dth doc assigned to a topic
	public int[][] n_d_zv, n_d_ze, n_d_zt;
	// dimension = K*V/T/E, each cell = # of times that word i assigned to a
	// topic k
	public double[][] n_zv_t, n_zt_t, n_ze_t, n_t_r;
	// K: total number of visual/text words assigned to topic j
	protected double[] sum_zv, sum_ze, sum_zt;

	// total number of visual/textual words in document i
	public int[] len_dv, len_dt, len_de;
	// total number of textual words in the corpus
	public int sum_text_words = 0;

	public double[][] p_zv_t, p_ze_t, p_zt_t, p_t_r; // phi
	public String p_zv_t_file, p_ze_t_file, p_zt_t_file;
	// compute perplexity every N steps
	public int trainStep = 10, testStep = 10; 

	public void initTrain(int K, int E, double alpha_v, double alpha_e,
			double beta_v, double beta_e, double gamma_t, double eta,
			String visual_file, String emotion_file, String text_file,
			int voc_v, int voc_e, int voc_t) {
		this.K = K;
		this.E = E;
		this.R = 2; // visual or emotional
		this.alpha_v = alpha_v;
		this.alpha_e = alpha_e;
		this.beta_v = beta_v;
		this.beta_e = beta_e;
		this.gamma_t = gamma_t;
		this.eta = eta;

		this.voc_v = voc_v;
		this.voc_e = voc_e;
		this.voc_t = voc_t;
		this.Vbeta = voc_v * beta_v;
		this.Ebeta = voc_e * beta_e;
		this.Tgamma = voc_t * gamma_t;
		this.Kalpha = K * alpha_v;
		this.Ealpha = E * alpha_e;

		this.visualFile = visual_file;
		this.textFile = text_file;
		this.emotionFile = emotion_file;
		this.visual_corpus = VELDAUtil.loadCorpus(visual_file);
		this.text_corpus = VELDAUtil.loadCorpus(text_file);
		this.emotion_corpus = VELDAUtil.loadCorpus(emotion_file);

		this.D = visual_corpus.size();
		System.out.println("Document size: " + D);
		System.out.println("Vocabulary size: visual=" + voc_v + " emotion="
				+ voc_e + " text=" + voc_t);

		// Dim = D*K, each cell = # of words belong to topic k in the dth doc.
		this.n_d_zv = new int[D][K];
		this.n_d_ze = new int[D][E];
		this.n_d_zt = new int[D][K + E];

		// n_z_t[k][i]: # of times that word i assigned to topic k
		this.n_zv_t = new double[K][voc_v];
		this.n_ze_t = new double[E][voc_e];
		this.n_zt_t = new double[K + E][voc_t];
		this.n_t_r = new double[voc_t][this.R];

		for (double[] row : n_zv_t)
			Arrays.fill(row, this.beta_v);
		for (double[] row : n_ze_t)
			Arrays.fill(row, this.beta_e);
		for (double[] row : n_zt_t)
			Arrays.fill(row, this.gamma_t);
		for (double[] row : n_t_r)
			Arrays.fill(row, this.eta);

		this.sum_zv = new double[K];
		Arrays.fill(sum_zv, this.Vbeta);
		this.sum_ze = new double[E];
		Arrays.fill(sum_ze, this.Ebeta);
		this.sum_zt = new double[K + E];
		Arrays.fill(sum_zt, this.Tgamma);

		this.len_dv = new int[D];
		this.len_de = new int[D];
		this.len_dt = new int[D];

		this.zv_d_n = new ArrayList<int[]>();
		this.ze_d_n = new ArrayList<int[]>();
		this.zt_d_n = new ArrayList<int[]>();

		for (int i = 0; i < D; i++) {
			int sum_vw = visual_corpus.get(i).length;
			int sum_tw = text_corpus.get(i).length;
			int sum_ew = emotion_corpus.get(i).length;
			len_dv[i] = sum_vw;
			len_de[i] = sum_ew;
			len_dt[i] = sum_tw;
			sum_text_words += sum_tw;

			zv_d_n.add(new int[sum_vw]);
			ze_d_n.add(new int[sum_ew]);
			zt_d_n.add(new int[sum_tw]);
		}
		initTopicLabel();
	}

	public void initTopicLabel() {
		System.out.print("Initing topic labels...");
		long start = System.currentTimeMillis();

		for (int d = 0; d < D; d++) {
			// visual topic
			int[] visual_doc = this.visual_corpus.get(d);
			int[] zv_n = this.zv_d_n.get(d);
			for (int n = 0; n < this.len_dv[d]; n++) {
				int t = visual_doc[n];
				int z = (int) (Math.random() * K);// [0, K - 1]
				zv_n[n] = z;
				this.n_d_zv[d][z] += 1;
				this.n_zv_t[z][t] += 1;
				this.sum_zv[z] += 1;
			}

			// emotional topic
			int[] emotion_doc = this.emotion_corpus.get(d);
			int[] ze_n = this.ze_d_n.get(d);
			for (int n = 0; n < this.len_de[d]; n++) {
				int t = emotion_doc[n];
				int e = (int) (Math.random() * E);// [0, E - 1]
				ze_n[n] = e;
				this.n_d_ze[d][e] += 1;
				this.n_ze_t[e][t] += 1;
				this.sum_ze[e] += 1;
			}

			// text topic
			int[] text_doc = this.text_corpus.get(d);
			int[] zt_n = this.zt_d_n.get(d);
			for (int n = 0; n < this.len_dt[d]; n++) {
				int t = text_doc[n];
				int z = (int) (Math.random() * (K + E));// [0, K+E-1]
				zt_n[n] = z;
				int r = relevance(z);
				this.n_t_r[t][r] += 1;
				this.n_zt_t[z][t] += 1;
				this.sum_zt[z] += 1;
				this.n_d_zt[d][z] += 1;
			}
		}

		long end = System.currentTimeMillis();
		System.out.println(" used " + (end - start) + "ms");
	}

	private int relevance(int z) {
		return z < this.K ? 0 : 1;
	}

	public void inference() {
		for (int d = 0; d < D; d++) {
			int[] n_zt = this.n_d_zt[d];

			/************** visual topic *************/
			int[] visual_doc = this.visual_corpus.get(d);
			int[] n_zv = this.n_d_zv[d];
			int[] zv_n = this.zv_d_n.get(d);
			for (int n = 0; n < this.len_dv[d]; n++) {
				int t = visual_doc[n];
				int z = zv_n[n];

				n_zv[z] -= 1;
				this.n_zv_t[z][t] -= 1;
				this.sum_zv[z] -= 1;

				double[] p_z = new double[K];
				for (int k = 0; k < K; k++) {
					double first = this.n_zv_t[k][t] / this.sum_zv[k];
					double second = n_zv[k] + this.alpha_v;
					double third = 1.0;
					if (n_zv[k] > 0) {
						third = Math.pow((n_zv[k] + 1.0) / n_zv[k], n_zt[k]);
					}
					p_z[k] = first * second * third;
				}
				int new_z = VELDAUtil.multinomial(p_z, K);
				zv_n[n] = new_z;

				n_zv[new_z] += 1;
				this.n_zv_t[new_z][t] += 1;
				this.sum_zv[new_z] += 1;
			}
			/************** end of visual topic *************/

			/************** emotion topic *************/
			int[] emotion_doc = this.emotion_corpus.get(d);
			int[] n_ze = this.n_d_ze[d];
			int[] ze_n = this.ze_d_n.get(d);
			for (int n = 0; n < this.len_de[d]; n++) {
				int t = emotion_doc[n];
				int z = ze_n[n];

				n_ze[z] -= 1;
				this.n_ze_t[z][t] -= 1;
				this.sum_ze[z] -= 1;

				double[] p_z = new double[E];
				for (int e = 0; e < E; e++) {
					double first = this.n_ze_t[e][t] / this.sum_ze[e];
					double second = n_ze[e] + this.alpha_e;
					double third = 1.0;
					if (n_ze[e] > 0) {
						third = Math.pow((n_ze[e] + 1.0) / n_ze[e],
								n_zt[K + e]);
					}
					p_z[e] = first * second * third;
				}
				int new_z = VELDAUtil.multinomial(p_z, E);
				ze_n[n] = new_z;

				n_ze[new_z] += 1;
				this.n_ze_t[new_z][t] += 1;
				this.sum_ze[new_z] += 1;
			}
			/************** end of emotion topic *************/

			/************** textual words *************/
			int[] text_doc = this.text_corpus.get(d);
			int[] zt_n = this.zt_d_n.get(d);
			for (int n = 0; n < this.len_dt[d]; n++) {
				int t = text_doc[n];
				int z = zt_n[n];
				int r = this.relevance(z);

				// discount the counters
				n_zt[z] -= 1;
				this.n_t_r[t][r] -= 1;
				this.sum_zt[z] -= 1;
				this.n_zt_t[z][t] -= 1;

				// compute distribution
				double[] p_z = new double[K + E];
				for (int k = 0; k < K; k++) {
					p_z[k] = (this.n_zt_t[k][t] / this.sum_zt[k]) * n_t_r[t][0]
							* n_zv[k] / this.len_dv[d];
				}
				for (int e = 0; e < E; e++) {
					p_z[e + K] = this.n_zt_t[e + K][t] / this.sum_zt[e + K]
							* n_t_r[t][1] * n_ze[e] / this.len_de[d];
				}

				int new_z = VELDAUtil.multinomial(p_z, K + E);
				int new_r = this.relevance(new_z);
				zt_n[n] = new_z;

				// increase the counters
				n_zt[new_z] += 1;
				this.n_t_r[t][new_r] += 1;
				this.sum_zt[new_z] += 1;
				this.n_zt_t[new_z][t] += 1;
			}
			/************** end of textual words *************/
		}
	}

	/*
	 * Apply the learned model on unseen new data (like testing).
	 * 
	 * This code is similar to the inference, except using the topic-word
	 * distribution (p_zv_t. p_ze_t and p_zt_t) from training.
	 */

	public void prediction() {
		for (int d = 0; d < D; d++) {
			int[] visual_doc = this.visual_corpus.get(d);
			int[] text_doc = this.text_corpus.get(d);
			int[] emotion_doc = this.emotion_corpus.get(d);

			int[] n_zv = this.n_d_zv[d];
			int[] n_ze = this.n_d_ze[d];
			int[] n_zt = this.n_d_zt[d];
			int[] zv_n = this.zv_d_n.get(d);
			int[] ze_n = this.ze_d_n.get(d);
			int[] zt_n = this.zt_d_n.get(d);

			/************** visual topic *************/
			for (int n = 0; n < this.len_dv[d]; n++) {
				int t = visual_doc[n];
				int z = zv_n[n];
				n_zv[z] -= 1;
				this.n_zv_t[z][t] -= 1;
				this.sum_zv[z] -= 1;

				double[] p_z = new double[K];
				for (int k = 0; k < K; k++) {
					double first = this.p_zv_t[k][t];
					double second = (n_zv[k] + this.alpha_v);
					double third = 1.0;
					if (n_zv[k] > 0) {
						third = Math.pow((n_zv[k] + 1.0) / n_zv[k], n_zt[k]);
					}
					p_z[k] = first * second * third;
				}

				int new_z = VELDAUtil.multinomial(p_z, K);

				zv_n[n] = new_z;
				n_zv[new_z] += 1;
				this.n_zv_t[new_z][t] += 1;
				this.sum_zv[new_z] += 1;
			}
			/************** end of visual topic *************/

			/************** emotion topic *************/
			for (int n = 0; n < this.len_de[d]; n++) {
				int t = emotion_doc[n];
				int z = ze_n[n];
				n_ze[z] -= 1;
				this.n_ze_t[z][t] -= 1;
				this.sum_ze[z] -= 1;

				double[] p_z = new double[E];
				for (int e = 0; e < E; e++) {
					double first = this.p_ze_t[e][t];
					double second = (n_ze[e] + this.alpha_e);
					double third = 1.0;
					if (n_ze[e] > 0) {
						third = Math.pow((n_ze[e] + 1.0) / n_ze[e],
								n_zt[K + e]);
					}
					p_z[e] = first * second * third;
				}

				int new_z = VELDAUtil.multinomial(p_z, E);
				ze_n[n] = new_z;

				n_ze[new_z] += 1;
				this.n_ze_t[new_z][t] += 1;
				this.sum_ze[new_z] += 1;
			}
			/************** end of emotion topic *************/

			/************** textual words *************/
			for (int n = 0; n < this.len_dt[d]; n++) {
				int t = text_doc[n];
				int z = zt_n[n];
				int r = this.relevance(z);

				// discount the counters
				n_zt[z] -= 1;
				this.n_t_r[t][r] -= 1;
				this.sum_zt[z] -= 1;
				this.n_zt_t[z][t] -= 1;

				// compute distribution
				double[] p_z = new double[K + E + 1];
				for (int k = 0; k < K; k++) {
					p_z[k] = this.p_zt_t[k][t] * this.p_t_r[t][0] * n_zv[k]
							/ this.len_dv[d];
				}
				for (int e = K; e < K + E; e++) {
					p_z[e] = this.p_zt_t[e][t] * this.p_t_r[t][1] * n_ze[e - K]
							/ this.len_de[d];
				}

				int new_z = VELDAUtil.multinomial(p_z, K + E);
				int new_r = this.relevance(new_z);
				zt_n[n] = new_z;

				// increase the counters
				n_zt[new_z] += 1;
				this.n_t_r[t][new_r] += 1;
				this.sum_zt[new_z] += 1;
				this.n_zt_t[new_z][t] += 1;
			}
			/************** end of text words *************/
		}
	}

	// theta: visual topic distribution for image.
	public double[][] p_d_zv() {
		double[][] p = new double[D][K];
		for (int d = 0; d < D; d++) {
			double normalizer = this.len_dv[d] + this.Kalpha;
			int[] n_zv = this.n_d_zv[d];
			for (int k = 0; k < K; k++) {
				p[d][k] = (n_zv[k] + this.alpha_v) / normalizer;
			}
		}
		return p;
	}

	// theta: emotion topic distribution for image.
	public double[][] p_d_ze() {
		double[][] p = new double[D][E];
		for (int d = 0; d < D; d++) {
			double normalizer = this.len_de[d] + this.Ealpha;
			int[] n_ze = this.n_d_ze[d];
			for (int e = 0; e < E; e++) {
				p[d][e] = (n_ze[e] + this.alpha_e) / normalizer;
			}
		}
		return p;
	}

	// phi:topic-visual_word distribution
	public double[][] p_zv_t() {
		double[][] p = new double[K][this.voc_v];
		for (int k = 0; k < K; k++) {
			for (int t = 0; t < this.voc_v; t++) {
				p[k][t] = this.n_zv_t[k][t] / this.sum_zv[k];
			}
		}
		return p;
	}

	// phi:topic-emotion_word distribution
	public double[][] p_ze_t() {
		double[][] p = new double[E][this.voc_e];
		for (int e = 0; e < E; e++) {
			for (int t = 0; t < this.voc_e; t++) {
				p[e][t] = this.n_ze_t[e][t] / this.sum_ze[e];
			}
		}
		return p;
	}

	public double[][] p_zt_t() {
		double[][] p = new double[K + E][this.voc_t];
		for (int k = 0; k < K + E; k++) {
			for (int t = 0; t < this.voc_t; t++) {
				p[k][t] = this.n_zt_t[k][t] / this.sum_zt[k];
			}
		}
		return p;
	}

	// probability of relevance type of each textual word
	public double[][] p_t_r() {
		double[][] p = new double[this.voc_t][R];
		for (int t = 0; t < this.voc_t; t++) {
			double sum = 0.0;
			for (int r = 0; r < this.R; r++) {
				sum += this.n_t_r[t][r];
			}
			for (int r = 0; r < this.R; r++) {
				p[t][r] = (this.n_t_r[t][r]) / sum;
			}
		}
		return p;
	}

	public double perplexity() {
		double[][] p_zt_t = this.p_zt_t();
		double[][] p_d_zv = this.p_d_zv();
		double[][] p_d_ze = this.p_d_ze();
		double[][] p_t_r = this.p_t_r();
		double log_per = 0;
		for (int d = 0; d < D; d++) {
			int[] text_doc = this.text_corpus.get(d);
			int doc_len = this.len_dt[d];
			for (int n = 0; n < doc_len; n++) {
				int t = text_doc[n];
				double[] p_zv = p_d_zv[d];
				double[] p_ze = p_d_ze[d];
				double[] p_r = p_t_r[t];

				double visual = 0.0;
				for (int k = 0; k < K; k++) {
					visual += p_zt_t[k][t] * p_zv[k];
				}

				double emotion = 0.0;
				for (int e = 0; e < E; e++) {
					emotion += p_zt_t[e + K][t] * p_ze[e];
				}

				double sum = p_r[0] * visual + p_r[1] * emotion;
				log_per -= Math.log(sum);
			}
		}
		return Math.exp(log_per / sum_text_words);
	}

	public double perplexityTest() {
		double[][] p_zt_t = this.p_zt_t;
		double[][] p_d_zv = this.p_d_zv();
		double[][] p_d_ze = this.p_d_ze();
		double[][] p_t_r = this.p_t_r;
		double log_per = 0;
		for (int d = 0; d < D; d++) {
			int[] text_doc = this.text_corpus.get(d);
			int doc_len = this.len_dt[d];
			for (int n = 0; n < doc_len; n++) {
				int t = text_doc[n];
				double[] p_zv = p_d_zv[d];
				double[] p_ze = p_d_ze[d];
				double[] p_r = p_t_r[t];

				double visual = 0.0;
				for (int k = 0; k < K; k++) {
					visual += p_zt_t[k][t] * p_zv[k];
				}

				double emotion = 0.0;
				for (int e = 0; e < E; e++) {
					emotion += p_zt_t[e + K][t] * p_ze[e];
				}

				double sum = p_r[0] * visual + p_r[1] * emotion;
				log_per -= Math.log(sum);
			}
		}
		return Math.exp(log_per / sum_text_words);
	}

	public void saveModel(String file_base, String mode) {
		File f = new File(file_base);
		if (!f.exists())
			f.mkdirs();
		VELDAUtil.writeDistribution(this.p_d_zv(),
				file_base + File.separator + "p_d_zv.txt");
		VELDAUtil.writeDistribution(this.p_d_ze(),
				file_base + File.separator + "p_d_ze.txt");
		if (mode.equals("train")) {
			VELDAUtil.writeDistribution(this.p_zv_t(),
					file_base + File.separator + "p_zv_t.txt");
			VELDAUtil.writeDistribution(this.p_ze_t(),
					file_base + File.separator + "p_ze_t.txt");
			VELDAUtil.writeDistribution(this.p_zt_t(),
					file_base + File.separator + "p_zt_t.txt");
			VELDAUtil.writeDistribution(this.p_t_r(),
					file_base + File.separator + "p_t_r.txt");
		}
	}

	public String configure(String mode) {
		if (mode.equals("train"))
			return "visual_file\t" + this.visualFile + "\n" + "emotion_file\t"
					+ this.emotionFile + "\n" + "text_file\t" + this.textFile
					+ "\n" + "K\t" + this.K + "\n" + "E\t" + this.E + "\n"
					+ "R\t" + this.R + "\n" + "alpha_v\t" + this.alpha_v + "\n"
					+ "alpha_e\t" + this.alpha_e + "\n" + "beta_v\t"
					+ this.beta_v + "\n" + "beta_e\t" + this.beta_e + "\n"
					+ "gamma_t\t" + this.gamma_t + "\n" + "eta\t" + this.eta
					+ "\n";
		else
			return "visual_file\t" + this.visualFile + "\n" + "emotion_file\t"
					+ this.emotionFile + "\n" + "text_file\t" + this.textFile
					+ "\n" + "K\t" + this.K + "\n" + "E\t" + this.E + "\n"
					+ "R\t" + this.R + "\n" + "alpha_v\t" + this.alpha_v + "\n"
					+ "alpha_e\t" + this.alpha_e + "\n" + "beta_v\t"
					+ this.beta_v + "\n" + "beta_e\t" + this.beta_e + "\n"
					+ "gamma_t\t" + this.gamma_t + "\n" + "eta\t" + this.eta
					+ "\n" + "p_zv_t_file\t" + p_zv_t_file + "\n"
					+ "p_ze_t_file\t" + p_ze_t_file + "\n" + "p_zt_t_file\t"
					+ p_zt_t_file + "\n";
	}

	public void initTest(String p_zv_t_file, String p_ze_t_file,
			String p_zt_t_file, String p_t_r_file, double alpha_v,
			double alpha_e, double beta_v, double beta_e, double gamma_t,
			double eta, String visual_file, String emotion_file,
			String text_file) {
		// load models
		this.p_zv_t_file = p_zv_t_file;
		this.p_zt_t_file = p_zt_t_file;
		this.p_ze_t_file = p_ze_t_file;
		this.p_zv_t = VELDAUtil.readDistribution(p_zv_t_file);
		this.p_ze_t = VELDAUtil.readDistribution(p_ze_t_file);
		this.p_zt_t = VELDAUtil.readDistribution(p_zt_t_file);
		this.p_t_r = VELDAUtil.readDistribution(p_t_r_file);

		int K = p_zv_t.length;
		int voc_v = p_zv_t[0].length;
		int E = p_ze_t.length;
		int voc_e = p_ze_t[0].length;
		int voc_t = p_zt_t[0].length;

		this.initTrain(K, E, alpha_v, alpha_e, beta_v, beta_e, gamma_t, eta,
				visual_file, emotion_file, text_file, voc_v, voc_e, voc_t);

	}

	public String train(String base, int iters, boolean save_intermediate_model)
			throws IOException, ParseException {
		long start = System.currentTimeMillis();
		System.out.println("----------------Training-------------");
		String base_path = base + File.separator + VELDAUtil.convertTime(start)
				+ "_K" + this.K + "_E" + this.E;
		File f = new File(base_path);
		if (!f.exists())
			f.mkdirs();

		PrintWriter pw = new PrintWriter(
				new FileWriter(base_path + File.separator + "log.txt"));
		pw.println(configure("train"));

		this.saveModel(base_path + File.separator + 0, "train");
		double perp = this.perplexity();
		pw.println("iter " + 0 + " " + perp);

		for (int i = 1; i <= iters; i++) {
			long t0 = System.currentTimeMillis();
			inference();
			long t1 = System.currentTimeMillis();
			System.out.println("iter " + i + " used " + (t1 - t0) + " ms");
			if (i % trainStep == 0) {
				double perpx = this.perplexity();
				pw.println("iter " + i + " " + perpx);
				// comment this to speed it up
				if (save_intermediate_model) {
					this.saveModel(base_path + File.separator + iters, "train");
				}
			}
			pw.flush();
		}

		// save the final model
		if (iters % trainStep != 0) {
			double perpx = this.perplexity();
			pw.println("iter " + iters + " " + perpx);
		}
		this.saveModel(base_path + File.separator + iters, "train");

		long end = System.currentTimeMillis();
		System.out.println("training used " + (end - start) / 1000 + "s");
		pw.println("training used " + (end - start) / 1000 + "s");
		pw.close();

		return base_path + File.separator + iters;
	}

	public String test(String base, int iters, boolean save_intermediate_model) throws IOException {
		long start = System.currentTimeMillis();

		System.out.println("----------------Testing-------------");
		String base_path = base + File.separator + VELDAUtil.convertTime(start);
		File f = new File(base_path);
		if (!f.exists())
			f.mkdirs();
		PrintWriter pw = new PrintWriter(
				new FileWriter(base_path + File.separator + "log.txt"));
		pw.println(configure("test"));

		this.saveModel(base_path + File.separator + 0, "test");
		double perp = this.perplexityTest();
		pw.println("iter " + 0 + " " + perp);

		for (int i = 1; i <= iters; i++) {
			long t0 = System.currentTimeMillis();
			prediction();
			long t1 = System.currentTimeMillis();
			System.out.println("iter " + i + " use " + (t1 - t0) + " ms");
			if (i % testStep == 0) {
				double perpx = this.perplexityTest();
				pw.println("iter " + i + " " + perpx);
				if (save_intermediate_model) {
					this.saveModel(base_path + File.separator + i, "test");
				}
			}
			pw.flush();
		}

		if (iters % testStep != 0) {
			double perpx = this.perplexityTest();
			pw.println("iter " + iters + " " + perpx);
		}
		this.saveModel(base_path + File.separator + iters, "test");

		long end = System.currentTimeMillis();
		System.out.println("testing used " + (end - start) / 1000 + " s");
		pw.println("Testing used " + (end - start) / 1000 + " s");
		pw.close();

		return base_path + File.separator + iters;
	}
}
