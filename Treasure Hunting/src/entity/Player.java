package entity;


import keyhandle.KeyHandle;
import main.GamePanel;
import objects.Axe;
import objects.Key;
import objects.SuperObject;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;

public class Player {
    public final int xScreen, yScreen;
    public int worldX, worldY;
    public int speed;
    private int number = 1;
    private int count = 0;
    private int changeDirt = 0;
    BufferedImage up1, up2, down1, down2, left1, left2, right1, right2;
    GamePanel gp;
    KeyHandle keyHandle;
    Rectangle solidArea;
    ArrayList<SuperObject> objects;
    boolean keyEnable;
    private int axeNumber;

    public Player(GamePanel gp) {
        this.gp = gp;
        keyHandle = gp.keyHandle;
        xScreen = (GamePanel.WIDTH - GamePanel.UNIT_SIZE) / 2;
        yScreen = (GamePanel.HEIGHT - GamePanel.UNIT_SIZE) / 2;
        setDefault();
        setImage();
    }

    private void setDefault() {
        objects = new ArrayList<>();
        solidArea = new Rectangle(14, 16, 20, 32);
        worldX = GamePanel.UNIT_SIZE * 3;
        worldY = GamePanel.UNIT_SIZE * 3;
        speed = 2;
        axeNumber = 2;
    }

    private void setImage() {
        try {
            up1 = ImageIO.read(new File("src/res/character/up1.png"));
            up2 = ImageIO.read(new File("src/res/character/up2.png"));
            down1 = ImageIO.read(new File("src/res/character/down1.png"));
            down2 = ImageIO.read(new File("src/res/character/down2.png"));
            left1 = ImageIO.read(new File("src/res/character/left1.png"));
            left2 = ImageIO.read(new File("src/res/character/left2.png"));
            right1 = ImageIO.read(new File("src/res/character/right1.png"));
            right2 = ImageIO.read(new File("src/res/character/right2.png"));
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    public void setAxeNumber() {
        axeNumber++;
    }

    public ArrayList<SuperObject> getObjects() {
        return objects;
    }

    public boolean isKeyEnable() {
        return keyEnable;
    }

    public void update() {
        if (keyHandle.up || keyHandle.down || keyHandle.left || keyHandle.right) {
            int playerLeft = (worldX + solidArea.x);
            int playerRight = (worldX + solidArea.x + solidArea.width);
            int playerTop = (worldY + solidArea.y);
            int playerBottom = (worldY + solidArea.y + solidArea.height);

            int playerLeftCol = playerLeft / GamePanel.UNIT_SIZE;
            int playerRightCol = playerRight / GamePanel.UNIT_SIZE;
            int playerTopRow = playerTop / GamePanel.UNIT_SIZE;
            int playerBottomRow = playerBottom / GamePanel.UNIT_SIZE;

            if (keyHandle.up) {
                playerTopRow = (playerTop - speed) / GamePanel.UNIT_SIZE;
                if (!gp.mapSetter.tiles[gp.map[playerTopRow][playerLeftCol]].isBarricade &&
                        !gp.mapSetter.tiles[gp.map[playerTopRow][playerRightCol]].isBarricade
                ) {
                    worldY -= speed;
                }
            } else if (keyHandle.down) {
                playerBottomRow = (playerBottom + speed) / GamePanel.UNIT_SIZE;
                if (!gp.mapSetter.tiles[gp.map[playerBottomRow][playerLeftCol]].isBarricade &&
                        !gp.mapSetter.tiles[gp.map[playerBottomRow][playerRightCol]].isBarricade
                ) {
                    worldY += speed;
                }
            }

            if (keyHandle.left) {
                playerLeftCol = (playerLeft - speed) / GamePanel.UNIT_SIZE;
                if (!gp.mapSetter.tiles[gp.map[playerTopRow][playerLeftCol]].isBarricade &&
                        !gp.mapSetter.tiles[gp.map[playerBottomRow][playerLeftCol]].isBarricade
                ) {
                    worldX -= speed;
                }
            } else if (keyHandle.right) {
                playerRightCol = (playerRight + speed) / GamePanel.UNIT_SIZE;
                if (!gp.mapSetter.tiles[gp.map[playerTopRow][playerRightCol]].isBarricade &&
                        !gp.mapSetter.tiles[gp.map[playerBottomRow][playerRightCol]].isBarricade
                ) {
                    worldX += speed;
                }
            }
            collectObjects();
            for(int i = 0; i<gp.getTraps().length; i++){
                gp.getTraps()[i].checkTrapped(this);
            }
        }

        count++;
        if (count >= 30) {
            count = 0;
            number = number % 2 + 1;
        }
        changeDirt++;
        if (changeDirt >= 200) {
            changeDirt = 0;
        }
    }

    private void collectObjects() {
        for (int i = 0; i < gp.object.length; i++) {
            if (gp.object[i] != null) {
                gp.object[i].checkCollection(this);
                if (gp.object[i].getIsCollected()) {
                    this.objects.add(gp.object[i]);
                    if (gp.object[i] instanceof Key) {
                        keyEnable = true;
                    }

                }
            }
        }
    }

    public void draw(Graphics2D g2) {
        BufferedImage image = null;
        switch (keyHandle.direction) {
            case "up":
                if (number == 1) image = up1;
                if (number == 2) image = up2;
                break;
            case "down":
                if (number == 1) image = down1;
                if (number == 2) image = down2;
                break;
            case "left":
                if (number == 1) image = left1;
                if (number == 2) image = left2;
                break;
            case "right":
                if (number == 1) image = right1;
                if (number == 2) image = right2;
                break;
        }
        g2.drawImage(image, xScreen, yScreen, GamePanel.UNIT_SIZE, GamePanel.UNIT_SIZE, null);
    }

    public int getChangeDirt() {
        return changeDirt;
    }

    public void useAxe(Axe axe) {
        int playerLeft = (worldX + solidArea.x);
        int playerRight = (worldX + solidArea.x + solidArea.width);
        int playerTop = (worldY + solidArea.y);
        int playerBottom = (worldY + solidArea.y + solidArea.height);

        int playerLeftCol = playerLeft / GamePanel.UNIT_SIZE;
        int playerRightCol = playerRight / GamePanel.UNIT_SIZE;
        int playerTopRow = playerTop / GamePanel.UNIT_SIZE;
        int playerBottomRow = playerBottom / GamePanel.UNIT_SIZE;
        switch (keyHandle.direction) {
            case "up" -> {
                playerTopRow = (playerTop - speed) / GamePanel.UNIT_SIZE;
                if (gp.mapSetter.tiles[gp.map[playerTopRow][playerLeftCol]].isBarricade) {
                    gp.map[playerTopRow][playerLeftCol] = 1;
                    for (int i = 0; i < gp.mapSetter.getLinkDoors().length; i++) {
                        if (playerTopRow == gp.mapSetter.getLinkDoors()[i].getGate().getFirst() &&
                                playerLeftCol == gp.mapSetter.getLinkDoors()[i].getGate().getSecond()
                        ) {
                            System.out.println(gp.map[gp.mapSetter.getLinkDoors()[i].getDoor().getFirst()][gp.mapSetter.getLinkDoors()[i].getDoor().getSecond()]);
                            gp.map[gp.mapSetter.getLinkDoors()[i].getDoor().getFirst()][gp.mapSetter.getLinkDoors()[i].getDoor().getSecond()] = 7;
                        }
                    }
                    axeNumber--;
                } else {
                    axe.setCapableOfChop(true);
                }
            }
            case "down" -> {
                playerBottomRow = (playerBottom + speed) / GamePanel.UNIT_SIZE;
                if (gp.mapSetter.tiles[gp.map[playerBottomRow][playerLeftCol]].isBarricade) {
                    gp.map[playerBottomRow][playerLeftCol] = 1;
                    for (int i = 0; i < gp.mapSetter.getLinkDoors().length; i++) {
                        if (playerBottomRow == gp.mapSetter.getLinkDoors()[i].getGate().getFirst() &&
                                playerLeftCol == gp.mapSetter.getLinkDoors()[i].getGate().getSecond()
                        ) {
                            System.out.println(gp.map[gp.mapSetter.getLinkDoors()[i].getDoor().getFirst()][gp.mapSetter.getLinkDoors()[i].getDoor().getSecond()]);
                            gp.map[gp.mapSetter.getLinkDoors()[i].getDoor().getFirst()][gp.mapSetter.getLinkDoors()[i].getDoor().getSecond()] = 7;
                        }
                    }
                    axeNumber--;
                } else {
                    axe.setCapableOfChop(true);
                }
            }
            case "left" -> {
                playerLeftCol = (playerLeft - speed) / GamePanel.UNIT_SIZE;
                if (gp.mapSetter.tiles[gp.map[playerBottomRow][playerLeftCol]].isBarricade) {
                    gp.map[playerTopRow][playerLeftCol] = 1;
                    for (int i = 0; i < gp.mapSetter.getLinkDoors().length; i++) {
                        if (playerTopRow == gp.mapSetter.getLinkDoors()[i].getGate().getFirst() &&
                                playerLeftCol == gp.mapSetter.getLinkDoors()[i].getGate().getSecond()
                        ) {
                            System.out.println(gp.map[gp.mapSetter.getLinkDoors()[i].getDoor().getFirst()][gp.mapSetter.getLinkDoors()[i].getDoor().getSecond()]);
                            gp.map[gp.mapSetter.getLinkDoors()[i].getDoor().getFirst()][gp.mapSetter.getLinkDoors()[i].getDoor().getSecond()] = 7;
                        }
                    }
                    axeNumber--;
                } else {
                    axe.setCapableOfChop(true);
                }
            }
            case "right" -> {
                playerRightCol = (playerRight + speed) / GamePanel.UNIT_SIZE;
                if (gp.mapSetter.tiles[gp.map[playerBottomRow][playerRightCol]].isBarricade) {
                    gp.map[playerTopRow][playerRightCol] = 1;
                    for (int i = 0; i < gp.mapSetter.getLinkDoors().length; i++) {
                        if (playerTopRow == gp.mapSetter.getLinkDoors()[i].getGate().getFirst() &&
                                playerRightCol == gp.mapSetter.getLinkDoors()[i].getGate().getSecond()
                        ) {
                            System.out.println(gp.map[gp.mapSetter.getLinkDoors()[i].getDoor().getFirst()][gp.mapSetter.getLinkDoors()[i].getDoor().getSecond()]);
                            gp.map[gp.mapSetter.getLinkDoors()[i].getDoor().getFirst()][gp.mapSetter.getLinkDoors()[i].getDoor().getSecond()] = 7;
                        }
                    }
                    axeNumber--;
                } else {
                    axe.setCapableOfChop(true);
                }
            }
        }
        if (!axe.isCapableOfChop()) {
            gp.unionFind.populate();
            if (!gp.unionFind.isConnected(3, 3, 4, 55)) {
                if (!gp.unionFind.isConnected(3, 3, 4, 42) || axeNumber < 1){
                    gp.setGameOver(true);
                }
            }
        }
    }
}
