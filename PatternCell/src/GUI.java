import javax.swing.*;
import javax.swing.border.Border;
import javax.swing.border.MatteBorder;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
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
  private TestPane grid;
  private JFrame frame;
  private JButton setButton;

  public static void main(String[] args) throws IOException
  {
    /*String s;

    Process p = Runtime.getRuntime().exec("python ../main.py");
    BufferedReader stdInput = new BufferedReader(new InputStreamReader(p.getInputStream()));
    while((s = stdInput.readLine()) != null) {
      System.out.println(s);
    }
    p.destroy();*/
    new GUI();
  }

  public GUI()
  {
    grid = new TestPane();

    frame = new JFrame("Grid");
    frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
    frame.setLayout(new BorderLayout());

    setButton = new JButton("What number is this?");
    setButton.setPreferredSize(new Dimension(160, 40));
    setButton.addActionListener(new ActionListener() {
      @Override
      public void actionPerformed(ActionEvent e) {
        grid.toString();
        grid.clear();
      }
    });

    frame.add(grid, BorderLayout.CENTER);
    frame.add(setButton, BorderLayout.PAGE_END);
    frame.pack();
    frame.setLocationRelativeTo(null);
    frame.setVisible(true);
  }

  private class TestPane extends JPanel
  {
    private int colCount;
    private int rowCount;
    private List<CellPane> cells = new ArrayList<>(colCount * rowCount);

    public TestPane()
    {
      colCount = 5;
      rowCount = 7;

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

    public void clear()
    {
      for(int i = 0; i < cells.size(); i++)
      {
        cells.get(i).clear();
      }
    }

    public List<CellPane> getCells()
    {
      return cells;
    }

    public List<Integer> getValues()
    {
      List<Integer> cellValues = new ArrayList(cells.size());

      for(int i = 0; i < cells.size(); i++)
      {
        cellValues.add(cells.get(i).getValue());
      }

      return cellValues;
    }

    public String toString() {
      String str = "";
      List<Integer> cellValues = this.getValues();

      for(int i = 0; i < cellValues.size(); i++)
      {
        str += cellValues.get(i);
        if(i < cellValues.size() - 1)
        {
          str += " ";
        }
      }

      return str;
    }
  }

  public class CellPane extends JPanel
  {
    private Color defaultBackground;

    public CellPane()
    {
      setPreferredSize(new Dimension(50, 50));
      defaultBackground = getBackground();

      addMouseListener(new MouseAdapter()
      {
        @Override
        public void mouseClicked(MouseEvent e)
        {
          if (getBackground().getBlue() == 0)
          {
            setBackground(defaultBackground);
          }
          else
          {
            setBackground(Color.BLACK);
          }
        }
      });
    }

    public void clear()
    {
      setBackground(defaultBackground);
    }

    public int getValue()
    {
      if(getBackground().getBlue() == 0) {
        return 1;
      }

      return 0;
    }

    public String toString()
    {
      return Integer.toString(this.getValue());
    }
  }
}