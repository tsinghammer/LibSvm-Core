using System;
using System.Collections.Generic;
using System.Text;
using System.Collections;
using System.Text.RegularExpressions;

namespace libsvm
{
    public class FeatureVector
    {
        #region Felder
        private Hashtable htVector;
        private int minTokenSize, maxTokens;
        private string[] trennzeichen;
        private Hashtable htVektorraum;
        private double[] vektor;
        private double label;
        #endregion

        #region Properties
        public double Label
        {
            get
            {
                return this.label;
            }
            set
            {
                this.label = value;
            }
        }

        public double[] Vektor
        {
            get
            {
                return this.vektor;
            }
            set
            {
                this.vektor = value;
            }
        }

        public Hashtable Vektorraum
        {
            get
            {
                return this.htVektorraum;
            }
            set
            {
                this.htVektorraum = value;
            }
        }

        public Hashtable HTVector
        {
            get
            {
                return this.htVector;
            }
        }

        public int MinTokenSize
        {
            get
            {
                return this.minTokenSize;
            }
            set
            {
                this.minTokenSize = value;
            }
        }

        public int MaxTokens
        {
            get
            {
                return this.maxTokens;
            }
            set
            {
                this.maxTokens = value;
            }
        }
        #endregion

        public FeatureVector(string text)
        {
            this.trennzeichen = new string[] { " ", "/", ",", ".", "\"", "'", ":", "\\", "?", "\n", "\r", "[", "]", "(", ")", "^", "    "};
            this.minTokenSize = 4;
            this.maxTokens = 10000;
            erzeugeHTVector(text);
        }

        /// <summary>
        /// Erzeugt eine Hashtable, die als Keys die Wörter dest Textes
        /// enthält und als Values der Häufigkeit im Text.
        /// </summary>
        /// <param name="text"></param>
        private void erzeugeHTVector(string text)
        {
            this.htVector = new Hashtable();

            // String Array Erzeugen
            string[] tokenArray = text.ToLower().Split(this.trennzeichen, StringSplitOptions.RemoveEmptyEntries);

            // Alles raus, was keine Miete zahlt
            List<string> tokens = new List<string>();

            foreach (string t in tokenArray)
            {
                if (t.Trim().Length >= this.minTokenSize)
                {
                    tokens.Add(Regex.Replace(t, @"[^\w\.@-]", ""));
                }
            }

            // Hashtable erstellen
            foreach (string t in tokens)
            {
                if (this.htVector.ContainsKey(t))
                {
                    this.htVector[t] = (double)this.htVector[t] + 1.0;
                }
                else
                {
                    this.htVector.Add(t, 1.0);
                }
            }
        }

        /// <summary>
        /// 
        /// </summary>
        public void ErzeugeVektor()
        {
            int index;
            int dimensionen = this.htVektorraum.Keys.Count;

            this.vektor = new double[dimensionen];

            foreach (string word in this.htVektorraum.Keys)
            {
                index = (int)this.htVektorraum[word];

                if (this.htVector.ContainsKey(word))
                {
                    this.vektor[index] = Math.Sqrt((double)this.htVector[word]);
                    //this.vektor[index] = 1.0;
                }
                else
                {
                    this.vektor[index] = 0.0;
                }
            }
        }
    }
}
