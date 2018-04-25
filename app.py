from flask import Flask, request, redirect, url_for,send_from_directory
from werkzeug.utils import secure_filename

import os
app = Flask(__name__)

UPLOAD_FOLDER= '/home/sebv/SebV/docker_flask/UPLOAD_FILES/'
ALLOWED_EXTENSIONS=set(['png'])


#############################TF PART####################

#1) charger le dictionnaire d entrainement
#2) rennomer les dossiers de validation et creer un nouveau dictionnaire correspondant
#3) load un dossier, le valider

import tensorflow as tf
import os.path
from resnet_model_NIR2 import Model
import numpy as np
import glob
from scipy import misc
from sklearn.metrics import *
import sys
from PIL import Image


global_path="/media/sebv/Data/IDENTIFICATION/results/64_classes_no_augmentation/"
model_path= global_path+"model.ckpt"
#image_path="/media/icar/Seagate Expansion Drive/testTF3/Validation/dossier_images"
#image_path="/home/sebv/SebV/datas/testTF3/Validation/dossier_images"
graph_path= global_path+"model.ckpt.meta"
IMAGE_WIDTH = 64
IMAGE_HEIGHT = 64
NUMBER_OF_CHANNELS = 3
BATCH_SIZE = 2
#TRAIN_FILE="home/data/sebV/Sebv/testTF3/Validation/dossier_images/1/*.png"

def load(saver, sess, ckpt_path):
    '''Load trained weights.

    Args:
      saver: TensorFlow saver object.
      sess: TensorFlow session.
      ckpt_path: path to checkpoint file with parameters.
    '''
    saver.restore(sess, ckpt_path)
    print("Restored model parameters from {}".format(ckpt_path))


def creation_batch(filename_queue,batch_size_arg):

	image_reader = tf.WholeFileReader()
	key, image_file = image_reader.read(filename_queue)
	S = tf.string_split([key],'/')
	length = tf.cast(S.dense_shape[1],tf.int32)


	# adjust constant value corresponding to your paths if you face issues. It should work for above format.
	
	####mode validation######
	#label = S.values[length-tf.constant(2,dtype=tf.int32)]
	######mode test sur image inconnue####
	label="1"
	label = tf.string_to_number(label,out_type=tf.int32)


	image = tf.image.decode_png(image_file,3,tf.uint16)

	image = tf.cast(image, tf.float32)
	image=tf.image.resize_images(image,[IMAGE_WIDTH,IMAGE_HEIGHT],)

	image = tf.reshape(image, [IMAGE_WIDTH,IMAGE_HEIGHT,NUMBER_OF_CHANNELS])

	label = tf.cast(label, tf.int64)



	print "toto"



	images_batch, labels_batch = tf.train.batch([image,label],batch_size=batch_size_arg,capacity=50000)
	labels_batch = tf.reshape(labels_batch, [batch_size_arg])

	return images_batch, labels_batch



def run(model_path,graph_path,dossier_source, dico):
	tf.reset_default_graph()
	 
	#dossiers =[s for s in os.listdir(dossier_source)]
	#tf.reset_default_graph()
	tr = tf.placeholder(tf.bool, name='training')
	#tf.reset_default_graph()
	mon_fichier = open(dico, "r")
        lignes= mon_fichier.readlines()
	dictionnaire={}
	for l in lignes:
		espece = l.split(":")[0]
		chiffre=l.split(":")[1]
		chiffre=chiffre.replace('\n','',1)
		chiffre=chiffre.replace('\r','',1)
		chiffre=chiffre.replace(' ','',1)

		dictionnaire[espece]=str(chiffre)



	dossiers=[]
	dossiers.append(dossier_source)
	print dossiers
	
	for d in dossiers:
			print ("###############"+ d)
			indice_espece=(d.split('/').__len__())-1

			
			dossier_courant=(d.split('/')[indice_espece])

			print dossier_courant
			model = Model(is_training=tr)
			if dossier_courant in dictionnaire.values():
					espece_en_cours= dictionnaire.keys()[dictionnaire.values().index(str(dossier_courant))]
			else:
				#print (dossier_courant +" not in dico")
				toto=0


			#print files
			batch_size_arg=1




			with tf.Session() as sess:

				#On cree les variables correspondant au model comme dans le training
				x = tf.placeholder(tf.float32, [None,IMAGE_WIDTH,IMAGE_HEIGHT,NUMBER_OF_CHANNELS], name='x-input2')
				y = tf.placeholder(tf.int64, [None], name='y-input2')

				train_file =d
				print train_file

				#####A VOIR ICI####
				filename_queue = tf.train.string_input_producer(tf.train.match_filenames_once(train_file))
				imgs_batch, labs_batch = creation_batch(filename_queue,batch_size_arg)


				#global_step = tf.train.get_or_create_global_step()

				logits = model.inference(x, n=1, reuse=tf.AUTO_REUSE) #fonction de prediction
				#loss = model.loss(logits=logits, labels=y)
				#accuracy = model.accuracy(logits, y)
				summary_op = tf.summary.merge_all()
				predictions=model.predictions(logits=logits)
		 		top_k_op = tf.nn.in_top_k(logits, y, 1)
				reader = tf.TextLineReader()
				key, value = reader.read(filename_queue)
				prediction2=tf.argmax(logits,1)
				#confusion=tf.contrib.metrics.confusion_matrix(y, prediction2)
				proba=tf.sigmoid(logits)		
				init=tf.global_variables_initializer()
				
				#On lance la session
				config = tf.ConfigProto()
				config.gpu_options.allow_growth = True
				sess = tf.Session(config=config)
				sess.run(init)
				
		        	sess.run(tf.local_variables_initializer())
				
				#On prepare le loader

				#variables_names = [v.name for v in tf.trainable_variables()]

				restore_var = tf.global_variables()
				loader = tf.train.Saver(var_list=restore_var)
				load(loader, sess,  tf.train.latest_checkpoint("/media/sebv/Data/IDENTIFICATION/results/64_classes_no_augmentation") )


				coord = tf.train.Coordinator()
				threads = tf.train.start_queue_runners(coord=coord, sess=sess)

				print"#########################"
				im_batch, lab_batch = sess.run([imgs_batch, labs_batch])
				feed_dict={x: im_batch, y: lab_batch, tr:False}
				for i in xrange(1):

						#print im_batch[0][0][0]
						#img_mat =Image.fromarray(im_batch,'RGB')
						#img_mat.show()


						preds = sess.run([prediction2],feed_dict=feed_dict)
						#preds =sess.run([predictions], feed_dict=feed_dict)
						#for i in range (len(preds))
						#	print

						#print preds[0]
						predict= preds[0].tolist()
						#predict_list=preds.tolist()
						print lab_batch,predict
						topito=sess.run([proba],feed_dict=feed_dict)
						#print topito
						top = topito[0].tolist()
						#print top
						#print ("Resultats Accuracy pour: "+str(accuracy_score(lab_batch, predict)))
						#print ("Resultats f1 pour: "+str(f1_score(lab_batch, predict, average=None)))
						#print ("Resultats recall pour: "+str(recall_score(lab_batch, predict, average=None)))
						#print ("Resultats precision pour: "+str(precision_score(lab_batch, predict, average=None)))
						#print ("Matrice de confusion: \n"+str(confusion_matrix(lab_batch, predict, labels=[0,1,2,3,4])))
						#print tab[0][0][0]
						
						#print "Resultats n "+str(i)+" pour "+ str(espece_en_cours)+" "+str(curr)
						dico={}
						j=0
						for i in top[0]:
							dico[i]=j
							j+=1
						sortie_proba=reversed(sorted(dico.keys()))

						inv_dico={v: k for k, v in dictionnaire.iteritems()}
						#print dico
						for t in sortie_proba:	
							if "'"+str(dico[t])+"'" in str(dictionnaire.values()):					print inv_dico[str(dico[t])], t

							else:
								print "'"+str(dico[t])+"'"+ " not in "+str(dictionnaire.values())

							
						#print sortie_proba

						print"#########################"
						###Ecriture des resultats####

						'''fichier_ecriture.write ("#######################"+  "\n")
						fichier_ecriture.write ("Espece: "+  espece_en_cours +" AKA "+  d+  "\n")
						fichier_ecriture.write ("Resultats Accuracy pour: "+str(accuracy_score(lab_batch, predict))+"\n")
						fichier_ecriture.write ("Resultats f1 pour: "+str(f1_score(lab_batch, predict, average=None))+"\n")
						fichier_ecriture.write ("Resultats recall pour: "+str(recall_score(lab_batch, predict, average=None))+"\n")
						fichier_ecriture.write ("Resultats precision pour: "+str(precision_score(lab_batch, predict, average=None))+"\n")
						fichier_ecriture.write ("Matrice de confusion: \n"+str(confusion_matrix(lab_batch, predict, labels=[0,1,2,3,4]))+"\n")
						fichier_ecriture.write ("#######################"+  "\n")'''

				coord.request_stop()
				coord.join(threads)


	return predict,lab_batch

def treat_dico(dico, dossier,cible):
	mon_fichier = open(dico, "r")
        lignes= mon_fichier.readlines()
	if os.path.exists(cible):
		print ("dico existant")

	else:
		mon_fichier_cible = open(cible, "w+")
		dictionnaire={} #nom_latin, chiffre
		for l in lignes:
			espece = l.split(":")[0]
			chiffre=l.split(":")[1]
			chiffre=chiffre.replace('\n','',1)
			dictionnaire[espece]=chiffre
		for (key, value) in dictionnaire.iteritems():
			print key,value

		dos = [s for s in os.listdir(dossier)]
		for d in dos:
			print d
			if dictionnaire.has_key(d):
				print dictionnaire[d]
				print ("Ecriture dico")
				mon_fichier_cible.write (d+":"+str(dictionnaire[d])+"\n")
	 	    		os.system("mv "+	dossier+"/"+d+" "+ dossier+"/"+str(dictionnaire[d]))
			else :
				print (d+" is already in dico")

    
#############################TF PART####################








app.config['UPLOAD_FOLDER']= UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
@app.route('/upload')
def upload_file_render():
   return render_template('upload.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


	
@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
   if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

	    #os.system("python demonstrateur/validate_on_picture_res.py "+UPLOAD_FOLDER+"/"+filename+" /media/sebv/Data/dataset/64_classes_no_augmentation/Metadata/dico")

	    #img_source = "/home/sebv/SebV/docker_flask/UPLOAD_FILES/Cephalopholis_argus_56x115826317_1718.png" #image a traite
	    img_source = UPLOAD_FOLDER+filename #image a traite
    	    dico_source = "/media/sebv/Data/dataset/64_classes_no_augmentation/Metadata/dico" #adresse du dico_source
    

    	    liste_resultat=[]
    	    liste_verite=[]

    	    predict, lab_batch=run(model_path,graph_path,img_source,dico_source)
	    
            return redirect(url_for('uploaded_file',
                                    filename=filename))
   return "c'est fait"



	
if __name__ == '__main__':
   app.run(debug = True)
