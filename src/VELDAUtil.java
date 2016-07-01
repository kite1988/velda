
import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.sql.Timestamp;
import java.util.ArrayList;
import java.util.Arrays;


public class VELDAUtil {

	public static ArrayList<int[]> loadCorpus(String file) {
		ArrayList<int[]> corpus = new ArrayList<int[]>();
		try {
			BufferedReader br = new BufferedReader(new FileReader(file));
			String line = null;
			while ((line = br.readLine()) != null) {
				line = line.trim();
				if (!line.isEmpty()) {
					String[] words = line.split(" ");
					int[] wordIds = new int[words.length];
					for (int i = 0; i < words.length; i++) {
						wordIds[i] = Integer.parseInt(words[i]);
					}
					corpus.add(wordIds);
				}
			}
			br.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
		return corpus;
	}

	public static void writeDistribution(double[][] p, String file) {
		try {
			PrintWriter pw = new PrintWriter(new FileWriter(file));
			int rows = p.length;
			int cols = p[0].length;
			for (int i = 0; i < rows; i++) {
				for (int j = 0; j < cols; j++) {
					pw.print(p[i][j] + "\t");
				}
				pw.println();
			}
			pw.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	public static double[][] readDistribution(String file) {
		double[][] p = null;
		ArrayList<double[]> pList = new ArrayList<double[]>();
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
					probs[i] = Double.parseDouble(items[i]);
				}
				pList.add(probs);
			}
			br.close();

			// Convert pList to standard double[][]
			int rows = pList.size();
			int cols = pList.get(0).length;
			p = new double[rows][cols];
			for (int i = 0; i < rows; i++) {
				p[i] = pList.get(i);
			}
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
		return p;
	}

	public static String convertTime(long time) {
		Timestamp current = new Timestamp(time);
		return current.toString().replaceAll(":", "-").replace(" ", "-").replace(".", "-");
	}

	// Do multinomial sampling via cummulative method
	public static int multinomial(double[] p_z, int K) {
		// cummulate multinomial parameters
		for (int k = 1; k < K; k++) {
			p_z[k] += p_z[k - 1];
		}
		// scaled sample because of unnormalized p[]
		double u = Math.random() * p_z[K - 1];

		int new_z = 0;
		for (; new_z < K; new_z++) {
			if (p_z[new_z] > u)
				break;
		}
		if (new_z == K)
			System.out.println(Arrays.toString(p_z));
		return new_z;
	}

	public static String extractBase(String base) {
		if (base.endsWith(File.separator))
			base = base.substring(0, base.length() - 1);
		int last_slash = base.lastIndexOf(File.separator);
		return base.substring(0, last_slash);
	}
	
}
