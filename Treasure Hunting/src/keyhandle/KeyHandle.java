package keyhandle;

import main.GamePanel;
import objects.Axe;
import java.awt.event.KeyEvent;
import java.awt.event.KeyListener;

public class KeyHandle implements KeyListener {
    GamePanel gp;
    public boolean up, down, left, right;
    public String direction = "down";
    public KeyHandle(GamePanel gp){
        this.gp = gp;
    }

    @Override
    public void keyTyped(KeyEvent e) {
        if (e.getKeyChar() == 'j') {
            for (int i = 0; i < gp.object.length; i++) {
                if (gp.object[i] != null && gp.object[i] instanceof Axe && ((Axe) gp.object[i]).isCapableOfChop()) {
                    ((Axe) gp.object[i]).chop(gp.player);
                }
            }
            System.out.println("Chopping tree");
        }
    }

    @Override
    public void keyPressed(KeyEvent e) {
        switch (e.getKeyCode()) {
            case KeyEvent.VK_W, KeyEvent.VK_UP -> {
                up = true;
                direction = "up";
                System.out.println(direction);
            }
            case KeyEvent.VK_S, KeyEvent.VK_DOWN -> {
                down = true;
                direction = "down";
                System.out.println(direction);
            }
            case KeyEvent.VK_A, KeyEvent.VK_LEFT -> {
                left = true;
                direction = "left";
                System.out.println(direction);
            }
            case KeyEvent.VK_D, KeyEvent.VK_RIGHT -> {
                right = true;
                direction = "right";
                System.out.println(direction);
            }
        }
    }

    @Override
    public void keyReleased(KeyEvent e) {
        switch (e.getKeyCode()) {
            case KeyEvent.VK_W, KeyEvent.VK_UP -> {
                up = false;
                //direction = "up";
            }
            case KeyEvent.VK_S, KeyEvent.VK_DOWN -> {
                down = false;
                //direction = "down";
            }
            case KeyEvent.VK_A, KeyEvent.VK_LEFT -> {
                left = false;
                ///direction = "left";
            }
            case KeyEvent.VK_D, KeyEvent.VK_RIGHT -> {
                right = false;
                //direction = "right";
            }
        }
    }
}
