package map;

import algorithm.AAsterisk;
import main.GamePanel;

import javax.imageio.ImageIO;
import java.awt.*;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;

public class MapSetter {
    public Tile[] tiles;
    GamePanel gp;
    private ArrayList<AAsterisk.Pair> starRoad;
    private LinkDoor[] linkDoors;
    private final Thread changeDirtThread = new Thread(this::changeImage);

    public MapSetter(GamePanel gp) {
        this.gp = gp;
        tiles = new Tile[20];
        setLinkDoors();
        setImageData();
        changeDirtThread.start();
    }

    public void setImageData() {
        try {
            tiles[0] = new Tile(ImageIO.read(new File("src/res/map/wall.png")), true);
            tiles[1] = new Tile(ImageIO.read(new File("src/res/map/snow.png")), false);
            tiles[2] = new Tile(ImageIO.read(new File("src/res/map/star1.png")), false);
            tiles[3] = new Tile(ImageIO.read(new File("src/res/map/water.png")), true);
            tiles[4] = new Tile(ImageIO.read(new File("src/res/map/tree.png")), true);
            tiles[5] = new Tile(ImageIO.read(new File("src/res/map/dirt.png")), false);
            tiles[6] = new Tile(ImageIO.read(new File("src/res/map/wallRoom.png")), true);
            tiles[7] = new Tile(ImageIO.read(new File("src/res/objects/door.png")), false);
            tiles[8] = new Tile(ImageIO.read(new File("src/res/objects/door.png")), true);
            tiles[9] = new Tile(ImageIO.read(new File("src/res/map/carpet.png")), false);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }
    private void setLinkDoors(){
        linkDoors = new LinkDoor[3];
        linkDoors[0] = new LinkDoor(new AAsterisk.Pair(8, 5), new AAsterisk.Pair(4, 53));
        linkDoors[1] = new LinkDoor(new AAsterisk.Pair(12, 5), new AAsterisk.Pair(6, 53));
        linkDoors[2] = new LinkDoor(new AAsterisk.Pair(16, 5), new AAsterisk.Pair(9, 53));
    }
    public LinkDoor[] getLinkDoors(){
        return linkDoors;
    }

    public void changeImage() {
        try {
            if (gp.player.getChangeDirt() == 100) {
                tiles[2].image = ImageIO.read(new File("src/res/map/star1.png"));
            } else if (gp.player.getChangeDirt() == 1) {
                tiles[2].image = ImageIO.read(new File("src/res/map/star2.png"));
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void draw(Graphics2D g2) {
        new Thread(this::changeImage).start();

        int worldCol = 0;
        int worldRow = 0;
        while (worldCol < GamePanel.MAX_COLUMN && worldRow < GamePanel.MAX_ROW) {
            int worldX = worldCol * GamePanel.UNIT_SIZE;
            int worldY = worldRow * GamePanel.UNIT_SIZE;
            int screenX = worldX - gp.player.worldX + gp.player.xScreen;
            int screenY = worldY - gp.player.worldY + gp.player.yScreen;

            if (worldX + GamePanel.UNIT_SIZE > gp.player.worldX - gp.player.xScreen
                    && worldX - GamePanel.UNIT_SIZE < gp.player.worldX + gp.player.xScreen
                    && worldY + GamePanel.UNIT_SIZE > gp.player.worldY - gp.player.yScreen
                    && worldY - GamePanel.UNIT_SIZE < gp.player.worldY + gp.player.yScreen
            ) {
                g2.drawImage(tiles[gp.map[worldRow][worldCol]].image, screenX, screenY, GamePanel.UNIT_SIZE, GamePanel.UNIT_SIZE, null);
            }
            worldCol++;

            if (worldCol == GamePanel.MAX_COLUMN) {
                worldRow++;
                worldCol = 0;
            }
        }
    }

    public void setStarRoad(ArrayList<AAsterisk.Pair> starRoad) {
        if (starRoad == null) {
            System.out.println("NO!!!!!!!!");
        } else {
            this.starRoad = starRoad;
            for (AAsterisk.Pair point : this.starRoad) {
                gp.map[point.getFirst()][point.getSecond()] = 2;
            }
        }
    }
}
