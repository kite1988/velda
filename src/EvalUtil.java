import java.lang.Math;
import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;

public class EvalUtil {

	/*
	 * Use term sensitive relevance distribution.
	 */
	public static void retrieval(String query_file, String visual_theta_file,
			String emotion_theta_file, String t_r_file, String text_phi_file,
			String result_file, String top_results) throws IOException {

		double[][] p_d_zv = EvalUtil.loadImageTheta(visual_theta_file);
		double[][] p_d_ze = EvalUtil.loadImageTheta(emotion_theta_file);
		double[][] p_zt_t = EvalUtil.readDistribution(text_phi_file);
		double[][] p_t_r = EvalUtil.readDistribution(t_r_file);

		ArrayList<int[]> query_list = EvalUtil.loadProcessedQuery(query_file);

		int N = query_list.size();
		int K = p_d_zv[0].length;
		int E = p_d_ze[0].length;

		System.out.println("----------------Evaluating-------------");
		System.out.println("Queries " + N);
		System.out.println("K " + K);
		System.out.println("E " + E);

		PrintWriter pw_top = new PrintWriter(new FileWriter(top_results));
		pw_top.println(
				"id,top1_id,top1_score,top2_id,top2_score,top3_id,top3_score,"
						+ "top4_id,top4_score,top5_id,top5_score,gold_id,gold_score,"
						+ "gold_rank,random_id,random_score");
		int[] gold_ranks = new int[N];
		int total_rank = 0;
		System.out.print("Computed queries: ");
		for (int q = 0; q < N; q++) { // query idx
			if (q % 100 == 0)
				System.out.println(" " + q);
			Score[] scores = new Score[N]; // scores for this query
			int[] query = query_list.get(q);
			for (int i = 0; i < N; i++) { // image idx
				scores[i] = computeScore(query, p_d_zv[i], p_d_ze[i], p_zt_t,
						p_t_r, K, E, i);
			}

			int gold_rank = EvalUtil.topResults(scores, q, N, pw_top);
			gold_ranks[q] = gold_rank;
			total_rank += gold_rank;
		}
		System.out.println();

		System.out.println(
				"Average ranks for ground truth: " + (double) (total_rank) / N);

		PrintWriter pw = new PrintWriter(new FileWriter(result_file));
		pw.println("Percent,Error Rate");
		double[] percents = new double[101];
		for (int i = 0; i < 101; i++) {
			percents[i] = i * 0.01;
		}

		for (int i = 0; i < percents.length; i++) {
			pw.println(percents[i] + ","
					+ EvalUtil.errorRate(gold_ranks, percents[i], N));

		}
		pw.close();
	}

	/*
	 * Compute the P(query|theta_image) with p_t_r
	 */
	public static Score computeScore(int[] query, double[] visual_theta,
			double[] emotion_theta, double[][] phi, double[][] p_t_r, int K,
			int E, int img_idx) {
		double score = 0.0;
		for (int n = 0; n < query.length; n++) {
			int w = query[n];
			double visual = 0.0;
			for (int k = 0; k < K; k++) {
				visual += phi[k][w] * visual_theta[k];
			}

			double emotion = 0.0;
			for (int e = 0; e < E; e++) {
				emotion += phi[e + K][w] * emotion_theta[e];
			}

			double sum = p_t_r[w][0] * visual + p_t_r[w][1] * emotion;
			score += Math.log(sum);
		}
		return new Score(img_idx, score);
	}

	/*
	 * Output the ID and the score for top five results, the ground truth, and a
	 * random image. Return the rank of ground truth image
	 */
	public static int topResults(Score[] scores, int query, int N,
			PrintWriter pw) {
		int rank = -1;
		Arrays.sort(scores);
		boolean find = false;
		pw.print(query + "\t");
		for (int i = 0; i < N; i++) {
			Score s = scores[i];
			if (s.id == query) {
				rank = i; // the ground truth idx is the same as query
				find = true;
			}
			if (i < 5) {
				pw.print(s.id + "," + s.score + ",");
			} else if (find && i >= 5) {
				Score gold = scores[rank];
				pw.print(gold.id + "," + gold.score + "," + rank + ",");
				int rdm_image = (int) (Math.random() * N);// From 0 to N - 1
				pw.println(rdm_image + "," + scores[rdm_image].score);
				pw.flush();
				return rank;
			}
		}
		return rank;
	}

	/*
	 * For light-weighted evaluation.
	 */
	public static double errorRate(int[] gold_ranks, double percent, int N) {
		int correct = 0;
		double threshold = N * percent;
		for (int gold_rank : gold_ranks) {
			if (gold_rank + 1 <= threshold)
				correct++;
		}
		return 1.0 - (double) correct / (double) N;
	}

	/*
	 * Load the query(testing) file that contains word ids.
	 */
	public static ArrayList<int[]> loadProcessedQuery(String test_file) {
		ArrayList<int[]> query_list = new ArrayList<int[]>();
		BufferedReader br;
		try {
			br = new BufferedReader(new FileReader(test_file));
			String line = null;
			while ((line = br.readLine()) != null) {
				line = line.trim();
				if (line.equals(""))
					continue;
				String[] items = line.split("\\s+");
				int length = items.length;
				int[] ids = new int[length];
				for (int i = 0; i < length; i++) {
					ids[i] = Integer.parseInt(items[i]);
				}
				query_list.add(ids);
			}
			br.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (NumberFormatException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
		return query_list;
	}

	/*
	 * Load a two dimensional probabilities into double[][].
	 */
	public static double[][] readDistribution(String file) {
		double[][] p = null;
		ArrayList<double[]> p_list = new ArrayList<double[]>();
		try {
			BufferedReader br = new BufferedReader(new FileReader(file));
			String line = null;
			while ((line = br.readLine()) != null) {
				line = line.trim();
				if (line.equals(""))
					continue;
				String[] items = line.split("\\s+");
				int length = items.length;
				double[] probs = new double[length];
				for (int i = 0; i < length; i++) {
					probs[i] = Float.parseFloat(items[i]);
				}
				p_list.add(probs);
			}
			br.close();

			// Convert p_list to standard double[][]
			int rows = p_list.size();
			int cols = p_list.get(0).length;
			p = new double[rows][cols];
			for (int i = 0; i < rows; i++) {
				p[i] = p_list.get(i);
			}
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
		return p;
	}

	/*
	 * Load testing image topic distribution as double[][].
	 */
	public static double[][] loadImageTheta(String file) {
		ArrayList<double[]> theta_list = new ArrayList<double[]>();
		BufferedReader br;
		try {
			br = new BufferedReader(new FileReader(file));
			String line = null;
			while ((line = br.readLine()) != null) {
				line = line.trim();
				if (line.equals(""))
					continue;
				String[] items = line.split("\\s+");
				int length = items.length;
				double[] probs = new double[length];
				for (int i = 0; i < length; i++) {
					probs[i] = Double.parseDouble(items[i]);
				}
				theta_list.add(probs);
			}
			br.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (NumberFormatException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}

		int row = theta_list.size();
		int col = theta_list.get(0).length;
		double[][] thetas = new double[row][col];
		for (int i = 0; i < row; i++) {
			thetas[i] = theta_list.get(i);
		}
		return thetas;
	}

}

class Score implements Comparable<Score> {
	public int id;
	public double score;

	public Score(int id, double score) {
		this.id = id;
		this.score = score;
	}

	public int compareTo(Score s) {
		if (this.score > s.score)
			return -1;
		else if (this.score < s.score)
			return 1;
		else
			return 0;
	}

}
