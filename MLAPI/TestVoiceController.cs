using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace MLAPI
{
    [Route("api/[controller]")]
    [ApiController]
    public class TestVoiceController : ControllerBase
    {
        [HttpGet]
        public ActionResult Index() {
            //Create MLContext
            MLContext mlContext = new MLContext();
            // Load Trained Model
            DataViewSchema predictionPipelineSchema;
            var ModelFile = Path.GetFullPath("emotionClassification.mlnet");
            
            ITransformer predictionPipeline = mlContext.Model.Load(ModelFile, out predictionPipelineSchema);
            // Create PredictionEngines
            PredictionEngine<VoiceModelInput, VoiceModelOutput> predictionEngine = mlContext.Model.CreatePredictionEngine<VoiceModelInput, VoiceModelOutput>(predictionPipeline);

            var lines = System.IO.File.ReadAllLines(Path.GetFullPath("test.csv"));

            List<bool> rightPrediction = new();
            foreach (var line in lines) // skip header
            {
                var parts = line.Split(',');


                float[] features = parts.Select(x => float.Parse(x)).ToArray();


                // Input Data
                VoiceModelInput inputData = new VoiceModelInput
                {
                    Col0 = features[0],
                    Col1 = features[1],
                    Col2 = features[2],
                    Col3 = features[3],
                    Col4 = features[4],
                    Col5 = features[5],
                    Col6 = features[6],
                    Col7 = features[7],
                    Col8 = features[8],
                    Col9 = features[9],
                    Col10 = features[10],
                    Col11 = features[11],
                    Col12 = features[12],
                    Col13 = features[13],
                    Col14 = features[14],
                    Col15 = features[15],
                    Col16 = features[16],
                    Col17 = features[17],
                    Col18 = features[18],
                    Col19 = features[19],
                    Col20 = features[20],
                    Col21 = features[21],
                    Col22 = features[22],
                    Col23 = features[23],
                    Col24 = features[24],
                    Col25 = features[25],
                };

                // Get Prediction
                VoiceModelOutput prediction = predictionEngine.Predict(inputData);
                Console.WriteLine($"{prediction.PredictedLabel} == {features[26]}");
                rightPrediction.Add(prediction.PredictedLabel == features[26]);
            }
            var x = rightPrediction.Count(r => r == true);
            var y = rightPrediction.Count();
            var rate = ( (float) x/ y );
            return Ok(new { rate = rate * 100 });
        }
    }

    


    public enum Emotion
    {
        neutral = 1,
        calm,
        happy,
        sad,
        angry,
        fearful,
        disgust,
        surprised,
    }

    public class VoiceModelInput
    {
        [LoadColumn(0)]
        [ColumnName(@"col0")]
        public float Col0 { get; set; }

        [LoadColumn(1)]
        [ColumnName(@"col1")]
        public float Col1 { get; set; }

        [LoadColumn(2)]
        [ColumnName(@"col2")]
        public float Col2 { get; set; }

        [LoadColumn(3)]
        [ColumnName(@"col3")]
        public float Col3 { get; set; }

        [LoadColumn(4)]
        [ColumnName(@"col4")]
        public float Col4 { get; set; }

        [LoadColumn(5)]
        [ColumnName(@"col5")]
        public float Col5 { get; set; }

        [LoadColumn(6)]
        [ColumnName(@"col6")]
        public float Col6 { get; set; }

        [LoadColumn(7)]
        [ColumnName(@"col7")]
        public float Col7 { get; set; }

        [LoadColumn(8)]
        [ColumnName(@"col8")]
        public float Col8 { get; set; }

        [LoadColumn(9)]
        [ColumnName(@"col9")]
        public float Col9 { get; set; }

        [LoadColumn(10)]
        [ColumnName(@"col10")]
        public float Col10 { get; set; }

        [LoadColumn(11)]
        [ColumnName(@"col11")]
        public float Col11 { get; set; }

        [LoadColumn(12)]
        [ColumnName(@"col12")]
        public float Col12 { get; set; }

        [LoadColumn(13)]
        [ColumnName(@"col13")]
        public float Col13 { get; set; }

        [LoadColumn(14)]
        [ColumnName(@"col14")]
        public float Col14 { get; set; }

        [LoadColumn(15)]
        [ColumnName(@"col15")]
        public float Col15 { get; set; }

        [LoadColumn(16)]
        [ColumnName(@"col16")]
        public float Col16 { get; set; }

        [LoadColumn(17)]
        [ColumnName(@"col17")]
        public float Col17 { get; set; }

        [LoadColumn(18)]
        [ColumnName(@"col18")]
        public float Col18 { get; set; }

        [LoadColumn(19)]
        [ColumnName(@"col19")]
        public float Col19 { get; set; }

        [LoadColumn(20)]
        [ColumnName(@"col20")]
        public float Col20 { get; set; }

        [LoadColumn(21)]
        [ColumnName(@"col21")]
        public float Col21 { get; set; }

        [LoadColumn(22)]
        [ColumnName(@"col22")]
        public float Col22 { get; set; }

        [LoadColumn(23)]
        [ColumnName(@"col23")]
        public float Col23 { get; set; }

        [LoadColumn(24)]
        [ColumnName(@"col24")]
        public float Col24 { get; set; }

        [LoadColumn(25)]
        [ColumnName(@"col25")]
        public float Col25 { get; set; }

        [LoadColumn(26)]
        [ColumnName(@"col26")]
        public float Col26 { get; set; }

    }


    /// <summary>
    /// model output class for EmotionClassification.
    /// </summary>
    /// 
    public class VoiceModelOutput
    {
        [ColumnName(@"Features")]
        public float[] Features { get; set; }

        [ColumnName(@"PredictedLabel")]
        public float PredictedLabel { get; set; }

        [ColumnName(@"Score")]
        public float[] Score { get; set; }

    }

}
