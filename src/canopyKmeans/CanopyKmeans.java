package canopyKmeans;

import java.io.IOException;
import java.util.*;

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.conf.*;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapred.*;
import org.apache.hadoop.util.*;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;

import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;


public class CanopyKmeans extends Configured implements Tool{
	/*
	 * Configured is a default implementation of the Configurable interface:
	 * setConf method retains a private instance variable to the passed Configuration object
	 * getConf() returns that reference
	 * 
	 * Tool is an extension of the Configurable interface
	 * providing an addition run(..) method
	 */

	public static class Point {
		private final double x;
		private final double y;

		public Point(String s) {
			String[] xandyComb = s.split(",");
			this.x = Double.parseDouble(xandyComb[0]);
			this.y = Double.parseDouble(xandyComb[1]);
		}

		public double getX() {
			return this.x;
		}

		public double getY() {
			return this.y;
		}

		public static double getRoughDistance(Point point1, Point point2) {
			double point1Sum = point1.getX() + point1.getY();
			double point2Sum = point2.getX() + point2.getY();
			double pointDiff = Math.abs(point1Sum - point2Sum);

			return pointDiff;
		}
		
		public static double getDistance(Point point1, Point point2) {
			double xDiff = point1.getX() - point2.getX();
			double yDiff = point1.getY() - point2.getY();
			double xDiffSquared = Math.pow(xDiff, 2);
			double yDiffSquared = Math.pow(yDiff, 2);

			return Math.sqrt(xDiffSquared + yDiffSquared);
		}

		public String toString() {
			return this.x + "," + this.y;
		}

	}

	public static class PointsMapperCanopy extends Mapper<LongWritable, Text, Text, Text> {

		public ArrayList<Point> centers = new ArrayList<>();

		@Override
		public void setup(Context context) throws IOException, InterruptedException {
		// only call once
			
			super.setup(context);
			Configuration conf = context.getConfiguration();
			Path centroids = new Path(conf.get("centroid.path"));
			FileSystem fs = FileSystem.get(conf);
			SequenceFile.Reader reader = new SequenceFile.Reader(fs, centroids, conf);

			// read centroids from the file
			Text key = new Text();
			IntWritable value = new IntWritable();
			while (reader.next(key, value)) {
				// dont care of value (0,0,0)
				centers.add(new Point(key.toString()));
			}
			reader.close();
		}

		@Override
		public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
		// for each point
			
			int trackId = 1;
			double setT = 10.0;

			Point currentPoint = new Point(value.toString());
			
			// if this point is within any T distance of centroids
			for (int i = 0; i < centers.size(); i++) {
				Point currentCenter = centers.get(i);
				double currentValue = Point.getRoughDistance(currentPoint, currentCenter);
				if (currentValue <= setT) 
				{
					context.write(new Text(Integer.toString(trackId)), new Text(currentPoint.toString()));
					break;
				}
				trackId++;
			}
			
			// if this point is not within any of current centroids
			if (trackId == centers.size())
			{
				context.write(new Text(Integer.toString(-1)), new Text(currentPoint.toString()));
			}
			
		}

		@Override
		public void cleanup(Context context) throws IOException, InterruptedException {

		}

	}
	
	public static class PointsReducerCanopy extends Reducer<Text, Text, Text, Text> {

		public static enum Counter {
			CONVERGED
		}

		public List<Point> newCenters = new ArrayList<>();

		@Override
		public void setup(Context context) {

		}

		@Override
		public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
			
			// pick whatever the first one of the group to be the new centroid
			String newPoint = values.iterator().next().toString();
			Point currentPoint = new Point(newPoint);
			this.newCenters.add(currentPoint);

			context.write(key, new Text(currentPoint.toString()));
		}

		@Override
		public void cleanup(Context context) throws IOException, InterruptedException {
			// delete the old centroids
			// write the new centroids
			super.setup(context);
			Configuration conf = context.getConfiguration();

			Path centroid_path = new Path(conf.get("centroid.path"));
			FileSystem fs = FileSystem.get(conf);

			if (fs.exists(centroid_path)) {
				fs.delete(centroid_path, true);
			}

			final SequenceFile.Writer centerWriter = SequenceFile.createWriter(fs, conf, centroid_path, Text.class,
					IntWritable.class);

			final IntWritable value = new IntWritable(0);

			for (Point newCenter : this.newCenters) {
				centerWriter.append(new Text(newCenter.toString()), value);
			}
			
			centerWriter.close();

		}

	}

	public static class PointsMapper extends Mapper<LongWritable, Text, Text, Text> {

		public ArrayList<Point> centers = new ArrayList<>();

		@Override
		public void setup(Context context) throws IOException, InterruptedException {

			super.setup(context);
			Configuration conf = context.getConfiguration();
			Path centroids = new Path(conf.get("centroid.path"));

			FileSystem fs = FileSystem.get(conf);

			SequenceFile.Reader reader = new SequenceFile.Reader(fs, centroids, conf);

			// read centroids from the file and store them in a centroids variable
			Text key = new Text();
			IntWritable value = new IntWritable();
			while (reader.next(key, value)) {
				// dont care of value (0,0,0)
				centers.add(new Point(key.toString()));
			}
			reader.close();

		}

		@Override
		public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {

			/*
			 * Mapper input: (each call is on one point)
			 * value -> a point
			 * centers -> a list of centers
			 * (x,y), (x,y), (x,y), (x,y), ...
			 * Mapper output: label and points
			 * [1, (x,y)], [2, (x,y)], [3, (x,y)], ...
			 */

			// input: key -> charater offset, value -> a point (in Text)
			Point currentPoint = new Point(value.toString());

			int index = 0;
			double currentMin = Double.MAX_VALUE;

			for (int i = 0; i < centers.size(); i++) {
				Point currentCenter = centers.get(i);
				double currentValue = Point.getDistance(currentPoint, currentCenter);
				if (currentValue < currentMin) {
					currentMin = currentValue;
					index = i;
				}
			}
			// emit key (centroid id/centroid) and value (point)
			context.write(new Text(Integer.toString(index)), new Text(currentPoint.toString()));
		}

		@Override
		public void cleanup(Context context) throws IOException, InterruptedException {

		}

	}

	public static class PointsReducer extends Reducer<Text, Text, Text, Text> {

		public static enum Counter {
			CONVERGED
		}

		// new_centroids (variable to store the new centroids
		public List<Point> newCenters = new ArrayList<>();

		@Override
		public void setup(Context context) {

		}

		@Override
		public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {

			/*
			 * Reducer input: labled points
			 * [1, ((x,y),(x,y),(x,y))...], [2, ((x,y),(x,y),(x,y))...], ....
			 * Reducer output: centriod
			 * [1, (x,y)], [2, (x,y)], [3, (x,y)], ...
			 */

			// Input: key -> centroid id/centroid , value -> list of points
			// calculate the new centroid
			double sumOfX = 0;
			double sumOfY = 0;
			int count = 0;
			while (values.iterator().hasNext()) {
				count++;
				String newPoint = values.iterator().next().toString();
				Point currentPoint = new Point(newPoint);
				sumOfX = sumOfX + currentPoint.getX();
				sumOfY = sumOfY + currentPoint.getY();
			}

			// new_centroids.add() (store updated cetroid in a variable)
			double newCenterX = sumOfX / count;
			double newCenterY = sumOfY / count;
			Point center = new Point(newCenterX + "," + newCenterY);
			this.newCenters.add(center);

			context.write(key, new Text(center.toString()));
		}

		@Override
		public void cleanup(Context context) throws IOException, InterruptedException {
			// delete the old centroids
			// write the new centroids
			super.setup(context);
			Configuration conf = context.getConfiguration();

			Path centroid_path = new Path(conf.get("centroid.path"));
			FileSystem fs = FileSystem.get(conf);

			if (fs.exists(centroid_path)) {
				fs.delete(centroid_path, true);
			}

			final SequenceFile.Writer centerWriter = SequenceFile.createWriter(fs, conf, centroid_path, Text.class,
					IntWritable.class);

			final IntWritable value = new IntWritable(0);

			for (Point newCenter : this.newCenters) {
				centerWriter.append(new Text(newCenter.toString()), value);
			}

			centerWriter.close();
		}

	}

	public static void main(String[] args) throws Exception {
		
		// setup

		Configuration conf = new Configuration();

		Path center_path = new Path("centroid/cen.seq");
		conf.set("centroid.path", center_path.toString());

		FileSystem fs = FileSystem.get(conf);

		if (fs.exists(center_path))
		{
			fs.delete(center_path, true);
		}

		// write initial points
		
		final SequenceFile.Writer centerWriter = SequenceFile.createWriter(fs, conf, center_path, Text.class,
				IntWritable.class);
		
		final IntWritable value = new IntWritable(0);
		// write the first point
		centerWriter.append(new Text("9.293805975483108,8.886169685374657"), value);

		centerWriter.close();	

		long startTime = System.currentTimeMillis();
		
		ToolRunner.run(conf, new CanopyKmeans(), args);
		
		long endTime = System.currentTimeMillis();
		long duration = (endTime - startTime)/1000;

		System.out.println("The total running time is " + duration + " seconds.");

		// read the centroid file and print the centroids (final result)

		SequenceFile.Reader reader = new SequenceFile.Reader(fs, center_path, conf);

		Text finalKey = new Text();
		IntWritable finalValue = new IntWritable();
		System.out.println("The final centroids are:");
		while (reader.next(finalKey, finalValue)) {
			// dont care of value (0,0,0)
			System.out.println(finalKey.toString());
		}
		reader.close();
		

	}
	
    @Override
    public int run(String[] args) throws Exception 
    {
		// config
		// job
		// set the job parameters
    	for (int i = 0; i <= 10; i++) {
    		Configuration conf = getConf();
    		Job job = Job.getInstance(conf, "Canopy");
    		FileSystem fs = FileSystem.get(conf);

    		job.setJarByClass(CanopyKmeans.class);
    		job.setMapperClass(PointsMapperCanopy.class);
    		job.setReducerClass(PointsReducerCanopy.class);

    		job.setOutputKeyClass(Text.class);
    		job.setOutputValueClass(Text.class);
    		job.setNumReduceTasks(1);
        
    		FileInputFormat.addInputPath(job, new Path(args[0]));
    		fs.delete(new Path(args[1]),true);
    		FileOutputFormat.setOutputPath(job, new Path(args[1]));
    		
    		job.waitForCompletion(true);
    	}
    	
    	for (int i = 0; i <= 10; i++) {
    		Configuration conf = getConf();
    		Job job = Job.getInstance(conf, "Kmeans");
    		FileSystem fs = FileSystem.get(conf);

    		job.setJarByClass(CanopyKmeans.class);
    		job.setMapperClass(PointsMapper.class);
    		job.setReducerClass(PointsReducer.class);

    		job.setOutputKeyClass(Text.class);
    		job.setOutputValueClass(Text.class);
    		job.setNumReduceTasks(1);
        
    		FileInputFormat.addInputPath(job, new Path(args[0]));
    		fs.delete(new Path(args[1]),true);
    		FileOutputFormat.setOutputPath(job, new Path(args[1]));
    		
    		job.waitForCompletion(true);
    	}
    	return 1;
        
    }

}
