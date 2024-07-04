package main;

import javax.swing.*;

public class GameFrame extends JFrame {
    GameFrame(){
        this.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        this.setTitle("Mini game");
        this.setResizable(true);
        GamePanel gamePanel = new GamePanel();
        gamePanel.setGf(this);
        this.add(gamePanel);

        this.pack();
        this.setLocationRelativeTo(null);
        this.setVisible(true);
    }
}
