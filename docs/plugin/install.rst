.. _install:

Installation
---------------

Napari is still considered alpha phase software and may not install correctly on the
first attempt, if that happens please open an issue `with us here <https://github.com/volume-em/empanada-napari/issues>`_.
Or reach out to the napari developers directly `here <https://github.com/napari/napari/issues>`_.

.. note::

  **Only Python 3.7, 3.8, 3.9 are supported, 3.10 and later are not.**

1. If not already installed, you can `install miniconda here <https://docs.conda.io/en/latest/miniconda.html>`_.

2. Download the correct installer for your OS (Mac, Linux, Windows).

3. After installing `conda`, open a new terminal or command prompt window.

4. Verify conda installed correctly with::

    $ conda --help

  .. note::
      If you get a "conda not found error" the most likely cause is that the path wasn't updated correctly. Try restarting
      the terminal or command prompt window. If that doesn't work then
      see `fixing conda path on Mac/Linux <https://stackoverflow.com/questions/35246386/conda-command-not-found>`_
      or `fixing conda path on Windows <https://stackoverflow.com/questions/44597662/conda-command-is-not-recognized-on-windows-10>`_.

5. If you've previously installed and used conda, it's recommended (but optional) to create a new virtual environment in order to avoid dependency conflicts::

    $ conda create -y -n empanada -c conda-forge python=3.9
    $ conda activate empanada

6. Install napari with pip::

    $ python -m pip install "napari[all]"

7. To verify installation, run::

    $ napari

For alternative and more detailed installation instructions, see the
`official napari installation tutorial <https://napari.org/tutorials/fundamentals/installation>`_.

From here the easiest way to install empanada-napari is directly in napari.

1. From the “Plugins” menu, select “Install/Uninstall Plugins...”.

.. image:: ../_static/plugin-menu.png
  :align: center
  :width: 200px
  :alt: Napari Plugin menu

2. In the resulting window that opens, where it says “Install by name/URL”, type "empanada-napari".

.. image:: ../_static/plugin-install-dialog.png
  :align: center
  :width: 500px
  :alt: Plugin installation dialog

3. Click the “Install” button next to the input bar.

If installation was successful you should see empanada-napari in the Plugins menu. If you don't, restart napari.

If you still don't see it, try installing the plugin with pip::

	$ pip install empanada-napari