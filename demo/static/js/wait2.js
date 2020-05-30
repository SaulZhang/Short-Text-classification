function ReadonlyText(objText)
{
	if (objText){

		objText.style.backgroundColor = "menu";

		objText.style.color = "black";

		objText.readOnly=true;
	}
}

function DisableElements(container,blnHidenButton)

{

	if (!container)

	return;

	var aEle;

	if (navigator.appName =="Microsoft Internet Explorer")  //IE

	{

		for (var i=0;i<container.all.length;i++)

		{

			aEle = container.all[i];

			tagName = aEle.tagName.toUpperCase();

			if ((tagName=="SELECT")||(tagName=="BUTTON"))

			{

				aEle.disabled = true;

				if(tagName=="BUTTON" && blnHidenButton)

				{

					aEle.style.display = "none";

				}

			}

			else if (tagName=="INPUT")

			{

				if (aEle.type.toUpperCase()!="HIDDEN")

				{

					if (aEle.type.toUpperCase()=="TEXT")

					{

						ReadonlyText(aEle);

					}

					else

					{

						aEle.disabled = true;

					}

				}

				if((aEle.type.toUpperCase()=="BUTTON"||aEle.type.toUpperCase()=="SUBMIT") && blnHidenButton)

				{

					aEle.style.display = "none";

				}

			}

			else if (tagName=="TEXTAREA")

			{

				ReadonlyText(aEle);

			}

		}

	}

	else

	{

		var aEle = container.getElementsByTagName("select");

		for (var i=0;i< aEle.length;i++)

		{

			aEle[i].disabled = true;

		}



		aEle = container.getElementsByTagName("button");

		for (var i=0;i< aEle.length;i++)

		{

			aEle[i].disabled = true;

		}



		aEle = container.getElementsByTagName("textarea");

		for (var i=0;i< aEle.length;i++)

		{

			ReadonlyText(aEle[i]);

		}



		aEle = container.getElementsByTagName("input");

		for (var i=0;i< aEle.length;i++)

		{

			if (aEle[i].type.toUpperCase()!="HIDDEN")

			{

				if (aEle[i].type.toUpperCase()=="TEXT")

				{

					ReadonlyText(aEle[i]);

				}

				else

				{

					aEle[i].disabled = true;

				}

			}

			if((aEle[i].type.toUpperCase()=="BUTTON"||aEle[i].type.toUpperCase()=="SUBMIT")&&blnHidenButton)

			{

				aEle[i].style.display = "none";

			}

		}

	}

}

function DisableLinkElement(oElement)

{

	if (!oElement)

		return;

	if (oElement.tagName.toUpperCase()=="A")

	{

		oElement.disabled = true;

		oElement.onclick = CancelEvent;

	}

}

function DisableLinkElements(container)

{

	if (!container)

		return;

	var aEle;

	if (navigator.appName =="Microsoft Internet Explorer")  //IE

	{

		for (var i=0;i<container.all.length;i++)

		{

			aEle = container.all[i];

			tagName = aEle.tagName.toUpperCase();

			if ((tagName=="A") && (aEle.id==""))

			{

				aEle.disabled = true;

				aEle.onclick = CancelEvent;

			}

		}

	}

	else

	{

		var aEle = container.getElementsByTagName("a");

		for (var i=0;i< aEle.length;i++)

		{

			if (aEle[i].id == "")

			{

				aEle[i].disabled = true;

				aEle[i].onclick = CancelEvent;

			}

		}

	}

}

function getElementsChild(formName,elementName,i)

{

	var objReturn;

	var lngLenghth=document.forms[formName].elements[elementName].length;

	lngLenghth=parseFloat(lngLenghth);

	if (lngLenghth + "" == "NaN")

	{

		objReturn = document.forms[formName].elements[elementName];

	}

	else

	{

		objReturn = document.forms[formName].elements[elementName][parseFloat(i)];

	}

	return objReturn;

}

