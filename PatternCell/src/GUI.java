import javax.swing.*;
import javax.swing.border.Border;
import javax.swing.border.MatteBorder;
import java.awt.*;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.List;
import java.util.ArrayList;
import java.lang.Runtime;

public class GUI
{
  public static void main(String[] args) throws IOException
  {
    String s = null;

    Process p = Runtime.getRuntime().exec("python ../main.py");
    BufferedReader stdInput = new BufferedReader(new InputStreamReader(p.getInputStream()));
    while((s = stdInput.readLine()) != null) {
      System.out.println(s);
    }
    p.destroy();
    new GUI();
  }

  public GUI()
  {
    JFrame frame = new JFrame("Grid");
    frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
    frame.setLayout(new BorderLayout());
    frame.add(new TestPane(), BorderLayout.CENTER);
    frame.add(new Classifier(), BorderLayout.PAGE_END);
    frame.pack();
    frame.setLocationRelativeTo(null);
    frame.setVisible(true);
  }

  private class TestPane extends JPanel
  {
    private int colCount = 5;
    private int rowCount = 7;
    private List<CellPane> cells = new ArrayList<>(colCount * rowCount);

    public TestPane()
    {
      init();
    }

    public TestPane(int row, int col)
    {
      rowCount = row > 0 ? row : 8;
      colCount = col > 0 ? col : 8;

      init();
    }

    private void init()
    {
      setLayout(new GridBagLayout());

      GridBagConstraints gbc = new GridBagConstraints();

      for(int row = 0; row < rowCount; row++)
      {
        for(int col = 0; col < colCount; col++)
        {
          gbc.gridx = col;
          gbc.gridy = row;

          cells.add(new CellPane());
          CellPane cellPane = cells.get(cells.size() - 1);

          Border border = null;
          if(row < rowCount - 1)
          {
            if(col < colCount - 1)
            {
              border = new MatteBorder(1, 1, 0, 0, Color.GRAY);
            }
            else
            {
              border = new MatteBorder(1, 1, 0, 1, Color.GRAY);
            }
          }
          else
          {
            if(col < colCount - 1)
            {
              border = new MatteBorder(1, 1, 1, 0, Color.GRAY);
            }
            else
            {
              border = new MatteBorder(1, 1, 1, 1, Color.GRAY);
            }
          }

          cellPane.setBorder(border);
          add(cellPane, gbc);
        }
      }
    }

    public List<CellPane> getCells()
    {
      return cells;
    }
  }

  public class CellPane extends JPanel
  {
    private Color defaultBackground;

    public CellPane()
    {
      setPreferredSize(new Dimension(50, 50));
      defaultBackground = getBackground();

      addMouseListener(new MouseAdapter() {
        @Override
        public void mouseClicked(MouseEvent e) {
          if (getBackground().getBlue() == 0) {
            setBackground(defaultBackground);
          } else {
            setBackground(Color.BLACK);
          }
        }
      });
    }

    public float getValue() {
      if(getBackground().getBlue() == 0) {
        return 0.5f;
      }

      return -0.5f;
    }
  }

  private class Classifier extends JPanel
  {
    JTextField textField;
    JButton setButton;

    public Classifier()
    {
      setLayout(new FlowLayout());

      textField = new JTextField();
      textField.setPreferredSize(new Dimension(140, 20));
      setButton = new JButton("Classify");
      setButton.setPreferredSize(new Dimension(80, 20));

      setLayout(new FlowLayout());
      add(textField);
      add(setButton);
    }
  }
}