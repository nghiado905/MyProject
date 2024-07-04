package main;

import algorithm.AAsterisk;
import algorithm.UnionFind;
import map.Trap;
import objects.Axe;
import entity.Player;
import objects.Chest;
import objects.Key;
import objects.SuperObject;
import keyhandle.KeyHandle;
import map.MapSetter;
import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.*;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;

public class GamePanel extends JPanel implements Runnable {
    public static final int UNIT_SIZE = 48;
    public static final int MAX_ROW = 50;
    public static final int MAX_COLUMN = 90;
    public static final int WIDTH = 600;
    public static final int HEIGHT = 600;
    public final int[][] map = new int[MAX_ROW][MAX_COLUMN];
    public final int FPS = 180;
    public UnionFind unionFind;
    public AAsterisk asterisk;
    public Player player;
    public MapSetter mapSetter;
    public KeyHandle keyHandle;
    public SuperObject[] object;
    public GameFrame gf;
    Thread thread;
    boolean gameOver;
    boolean gameWin;
    Trap[] traps;

    public GamePanel() {
        getData();

        object = new SuperObject[10];
        setObject();

        traps = new Trap[2];
        setTraps();

        keyHandle = new KeyHandle(this);
        player = new Player(this);
        mapSetter = new MapSetter(this);

        unionFind = new UnionFind(this);
        asterisk = new AAsterisk(this);

        this.setPreferredSize(new Dimension(WIDTH, HEIGHT));
        this.setBackground(Color.CYAN);
        this.setFocusable(true);
        this.addKeyListener(keyHandle);

        startThread();
    }

    public void startThread() {
        thread = new Thread(this);
        thread.start();
    }

    @Override
    public void run() {
        double drawInterval = (double) 1000000000 / FPS;
        double realTime = 0;
        long lastTime = System.nanoTime();
        long crrTime;

        while (thread != null) {
            crrTime = System.nanoTime();
            realTime += (double) (crrTime - lastTime) / drawInterval;
            lastTime = crrTime;

            if (realTime >= 1) {
                update();
                repaint();

                realTime--;
            }
        }
    }

    public void getData() {
        try {
            BufferedReader reader = new BufferedReader(new FileReader("src/res/mapData/testMap.txt"));
            String line;
            int row = 0;
            while ((line = reader.readLine()) != null) {
                String[] data = line.split(" ");
                for (int i = 0; i < MAX_COLUMN; i++) {
                    map[row][i] = Integer.parseInt(data[i]);
                }
                row++;
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public void setObject() {
        object[0] = new Axe(this);
        object[4] = new Axe(this);
        ((Axe) object[4]).changeInformation();

        object[1] = new Chest(this);

        object[2] = new Chest(this);
        ((Chest) object[2]).changeInformation("suggestion", 66, 4, false, false);


        object[3] = new Key(this);
    }

    private void setTraps() {
        traps[0] = new Trap(this, 32, 21);
        traps[1] = new Trap(this, 31, 24);
    }

    public Trap[] getTraps() {
        return traps;
    }

    public void paintComponent(Graphics g) {
        super.paintComponent(g);
        draw(g);
    }

    public void draw(Graphics g) {
        Graphics2D g2 = (Graphics2D) g;

        if (gameOver) {
            this.setBackground(Color.BLACK);
            drawGameOver(g2);
            thread.interrupt();
        } else if (gameWin) {
            drawWin(g2);
            thread.interrupt();
        } else {
            mapSetter.draw(g2);
            for (SuperObject superObject : object) {
                if (superObject != null) {
                    superObject.draw(g2);
                }
            }
            for (Trap trap : traps) {
                if (trap != null)
                    trap.draw(g2);
                else {
                    System.out.println("Null!!!!!");
                }
            }
            player.draw(g2);
        }
        g2.dispose();
    }

    public void drawGameOver(Graphics2D g2) {
        this.setBackground(Color.WHITE);
        g2.setColor(Color.RED);
        g2.setFont(new Font("Time new roman", Font.BOLD, 75));
        FontMetrics metrics = getFontMetrics(g2.getFont());
        g2.drawString("GAME OVER", (WIDTH - metrics.stringWidth("GAME OVER")) / 2, (HEIGHT / 2));
    }

    public void drawWin(Graphics2D g2) {
        try {
            g2.drawImage(ImageIO.read(new File("src/res/gameEnd/win.png")), 0, 0, WIDTH, HEIGHT, null);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    public void update() {
        player.update();
    }

    public void setGameOver(boolean gameOver) {
        this.gameOver = gameOver;
    }

    public void setGameWin(boolean gameWin) {
        this.gameWin = gameWin;
    }

    public void setGf(GameFrame gf) {
        this.gf = gf;
    }
}
