import java.util.Scanner;

public class DistanceVectorRouting {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);

        System.out.print("Enter number of nodes: ");
        int n = sc.nextInt();

        int[][] d = new int[n][n];
        int[][] next = new int[n][n];

        System.out.println("Enter cost matrix:");
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                d[i][j] = sc.nextInt();
                next[i][j] = j;
            }
        }

        // Distance update (k = intermediate, i = source, j = destination)
        for (int k = 0; k < n; k++) {
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    if (d[i][k] + d[k][j] < d[i][j]) {
                        d[i][j] = d[i][k] + d[k][j];
                        next[i][j] = next[i][k];
                    }
                }
            }
        }

        // Print routing tables
        for (int i = 0; i < n; i++) {
            System.out.println("\nRouting table for Node " + i);
            System.out.println("Dest\tNextHop\tCost");
            for (int j = 0; j < n; j++) {
                System.out.println(j + "\t" + next[i][j] + "\t" + d[i][j]);
            }
        }

        // Print shortest path
        System.out.print("\nEnter client and server node: ");
        int src = sc.nextInt();
        int dest = sc.nextInt();

        System.out.print("Path: " + src);
        int curr = src;
        while (curr != dest) {
            curr = next[curr][dest];
            System.out.print(" -> " + curr);
        }
        System.out.println("\nTotal cost: " + d[src][dest]);
    }
}
