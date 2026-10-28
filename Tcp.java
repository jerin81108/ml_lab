import java.io.*;
import java.net.*;

public class ChatServer {
    public static void main(String[] args) throws Exception {

        ServerSocket ss = new ServerSocket(12345);
        System.out.println("Server waiting for client...");

        Socket s = ss.accept();
        System.out.println("Client connected!");

        BufferedReader in = new BufferedReader(
            new InputStreamReader(s.getInputStream()));

        PrintWriter out = new PrintWriter(
            s.getOutputStream(), true);

        BufferedReader keyboard = new BufferedReader(
            new InputStreamReader(System.in));

        String msg, reply;

        while (true) {
            msg = in.readLine();

            if (msg == null || msg.equalsIgnoreCase("exit")) {
                break;
            }

            System.out.println("Client: " + msg);

            System.out.print("Server: ");
            reply = keyboard.readLine();

            out.println(reply);

            if (reply.equalsIgnoreCase("exit")) {
                break;
            }
        }

        s.close();
        ss.close();

        System.out.println("Chat ended.");
    }
}








import java.io.*;
import java.net.*;

public class ChatClient {
    public static void main(String[] args) throws Exception {

        Socket s = new Socket("localhost", 12345);
        System.out.println("Connected to server!");

        BufferedReader in = new BufferedReader(
            new InputStreamReader(s.getInputStream()));

        PrintWriter out = new PrintWriter(
            s.getOutputStream(), true);

        BufferedReader keyboard = new BufferedReader(
            new InputStreamReader(System.in));

        String msg, reply;

        while (true) {
            System.out.print("Client: ");
            msg = keyboard.readLine();

            out.println(msg);

            if (msg.equalsIgnoreCase("exit")) {
                break;
            }

            reply = in.readLine();

            if (reply == null) {
                break;
            }

            System.out.println("Server: " + reply);

            if (reply.equalsIgnoreCase("exit")) {
                break;
            }
        }

        s.close();
        System.out.println("Chat ended.");
    }
}
