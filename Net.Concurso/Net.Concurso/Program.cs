using System;
using System.IO;
using System.Text;
using UglyToad.PdfPig;

class Program
{
    static void Main()
    {
        Console.Write("Enter the path of the PDF file: ");
        string pdfPath = Console.ReadLine();

        if (File.Exists(pdfPath))
        {
            try
            {
                string extractedText = ExtractTextFromPdf(pdfPath);
                string textFilePath = Path.ChangeExtension(pdfPath, ".txt");
                File.WriteAllText(textFilePath, extractedText);

                Console.WriteLine($"\nExtracted text saved to: {textFilePath}");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error reading PDF: {ex.Message}");
            }
        }
        else
        {
            Console.WriteLine("File does not exist.");
        }
    }

    static string ExtractTextFromPdf(string filePath)
    {
        using var document = PdfDocument.Open(filePath);
        StringBuilder textBuilder = new StringBuilder();

        foreach (var page in document.GetPages())
        {
            textBuilder.AppendLine(page.Text);
        }

        return textBuilder.ToString();
    }
}
