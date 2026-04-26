namespace Optical_Recognition_Pattern
{
    partial class Form1
    {
        /// <summary>
        ///  Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        ///  Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        /// <summary>
        ///  Required method for Designer support - do not modify
        ///  the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            pictureBox1 = new PictureBox();
            lblResult = new Label();
            btnPredict = new Button();
            btnClear = new Button();
            label1 = new Label();
            acc = new Label();
            ((System.ComponentModel.ISupportInitialize)pictureBox1).BeginInit();
            SuspendLayout();
            // 
            // pictureBox1
            // 
            pictureBox1.Location = new Point(256, 62);
            pictureBox1.Name = "pictureBox1";
            pictureBox1.Size = new Size(280, 280);
            pictureBox1.TabIndex = 0;
            pictureBox1.TabStop = false;
            pictureBox1.MouseMove += pictureBox1_MouseMove_1;
            // 
            // lblResult
            // 
            lblResult.AutoSize = true;
            lblResult.Font = new Font("Segoe UI", 14F);
            lblResult.Location = new Point(38, 380);
            lblResult.Name = "lblResult";
            lblResult.Size = new Size(125, 32);
            lblResult.TabIndex = 1;
            lblResult.Text = "Результат:";
            // 
            // btnPredict
            // 
            btnPredict.Location = new Point(641, 372);
            btnPredict.Name = "btnPredict";
            btnPredict.Size = new Size(119, 55);
            btnPredict.TabIndex = 2;
            btnPredict.Text = "Передбачити";
            btnPredict.UseVisualStyleBackColor = true;
            btnPredict.Click += btnPredict_Click_1;
            // 
            // btnClear
            // 
            btnClear.Location = new Point(641, 290);
            btnClear.Name = "btnClear";
            btnClear.Size = new Size(119, 52);
            btnClear.TabIndex = 3;
            btnClear.Text = "Очистити";
            btnClear.UseVisualStyleBackColor = true;
            btnClear.Click += btnClear_Click_1;
            // 
            // label1
            // 
            label1.AutoSize = true;
            label1.Location = new Point(38, 121);
            label1.Name = "label1";
            label1.Size = new Size(0, 20);
            label1.TabIndex = 4;
            // 
            // acc
            // 
            acc.AutoSize = true;
            acc.Location = new Point(26, 18);
            acc.Name = "acc";
            acc.Size = new Size(120, 20);
            acc.TabIndex = 5;
            acc.Text = "Точність моделі";
            // 
            // Form1
            // 
            AutoScaleDimensions = new SizeF(8F, 20F);
            AutoScaleMode = AutoScaleMode.Font;
            ClientSize = new Size(800, 450);
            Controls.Add(acc);
            Controls.Add(label1);
            Controls.Add(btnClear);
            Controls.Add(btnPredict);
            Controls.Add(lblResult);
            Controls.Add(pictureBox1);
            Name = "Form1";
            Text = "Form1";
            Load += Form1_Load;
            ((System.ComponentModel.ISupportInitialize)pictureBox1).EndInit();
            ResumeLayout(false);
            PerformLayout();
        }

        #endregion

        private PictureBox pictureBox1;
        private Label lblResult;
        private Button btnPredict;
        private Button btnClear;
        private Label label1;
        private Label acc;
    }
}
